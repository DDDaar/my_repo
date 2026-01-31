import os
import torch
from torch_npu.contrib import transfer_to_npu # 若非 Ascend NPU 可注释
import torch.distributed as dist
import argparse
import random
import numpy as np
from tqdm import tqdm
from torch.utils.data import DataLoader, DistributedSampler
from transformers import Qwen2_5_VLProcessor
import deepspeed
import wandb

from config.configuration_latent import LatentConfig
from model.modeling_latent_qwen import LatentReasoningQwen
from data.dataset_latent import LatentReasoningDataset, collate_fn

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--local_rank", type=int, default=int(os.environ.get("LOCAL_RANK", -1)))
    parser.add_argument("--wandb_project", type=str, default="latent_vbc_full")
    parser.add_argument("--wandb_run_name", type=str, default="run_autoregressive")
    parser.add_argument("--wandb_offline", action="store_true")
    parser = deepspeed.add_config_arguments(parser)
    return parser.parse_args()

def main():
    args = parse_args()

    # 分布式初始化
    if args.local_rank != -1:
        torch.cuda.set_device(args.local_rank)
        # 注意: NVIDIA GPU 请用 "nccl", Ascend NPU 用 "hccl"
        dist.init_process_group(backend="hccl") 

    # 加载配置与种子
    config = LatentConfig.load("./config/config.yaml")

    if len(config.train_datasets) == 1:
        # 单数据集模式
        raw_ds_name = config.train_datasets[0].name
        # 将 "derek-thomas/ScienceQA" 转换为 "derek-thomas_ScienceQA"
        dataset_dir_name = raw_ds_name.replace("/", "_")
    else:
        # 混合数据集模式
        dataset_dir_name = "mixed_datasets"
    print(f"--- Checkpoint Output Dir: ./checkpoints/{dataset_dir_name} ---")


    set_seed(config.seed) 

    if args.local_rank <= 0:
        print(f"--- Training Configuration ---")
        print(f"Seed: {config.seed}")
        print(f"Extract Prefix: '{config.extract_text_prefix}'")
        print(f"VBC Enabled: Lambda={config.lambda_vbc}, Margin={config.vbc_margin}")
        print(f"------------------------------")
        
        wandb.init(
            project=args.wandb_project,
            name=args.wandb_run_name,
            mode="offline" if args.wandb_offline else "online",
            config=config.__dict__
        )

    # Processor & Tokenizer
    processor = Qwen2_5_VLProcessor.from_pretrained(config.base_model)
    
    # 注册所有特殊 Token
    new_tokens = [config.extract_token] + [stage.token for stage in config.stages]
    processor.tokenizer.add_special_tokens({"additional_special_tokens": new_tokens})

    # 更新 Config ID
    config.extract_token_id = processor.tokenizer.convert_tokens_to_ids(config.extract_token)
    for stage in config.stages:
        stage.token_id = processor.tokenizer.convert_tokens_to_ids(stage.token)

    # Model
    model = LatentReasoningQwen(config)
    model.base_model.resize_token_embeddings(len(processor.tokenizer))
    model.len_tokenizer = len(processor.tokenizer)

    # Dataset & DataLoader
    train_dataset = LatentReasoningDataset(processor, config)
    sampler = DistributedSampler(train_dataset) if args.local_rank != -1 else None
    
    g = torch.Generator()
    g.manual_seed(config.seed)

    dataloader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        sampler=sampler,
        collate_fn=collate_fn,
        num_workers=4,
        pin_memory=True,
        worker_init_fn=seed_worker,
        generator=g
    )

    # DeepSpeed Init
    model_engine, optimizer, _, _ = deepspeed.initialize(
        args=args,
        model=model,
        model_parameters=[p for p in model.parameters() if p.requires_grad]
    )

    # Training Loop
    global_step = 0
    total_steps = len(dataloader) * config.epochs
    
    for epoch in range(config.epochs):
        if sampler: sampler.set_epoch(epoch)
        
        pbar = tqdm(enumerate(dataloader), total=len(dataloader), disable=(args.local_rank > 0))
        
        for step, batch in pbar:
            global_step += 1
            
            # Move to device
            batch_gpu = {}
            for k, v in batch.items():
                if isinstance(v, torch.Tensor):
                    batch_gpu[k] = v.to(model_engine.device)
                elif isinstance(v, dict): 
                    batch_gpu[k] = {sk: sv.to(model_engine.device, dtype=torch.bfloat16) for sk, sv in v.items()}
                else:
                    batch_gpu[k] = v

            outputs = model_engine(**batch_gpu)
            loss = outputs["loss"]
            
            model_engine.backward(loss)
            model_engine.step()

            if args.local_rank <= 0:
                wandb.log({
                    "train/loss": loss.item(),
                    "train/sft_ans": outputs["sft_loss"].item(),
                    "train/latent_ce": outputs["latent_ce"].item(),
                    "train/mse": outputs["mse_loss"].item(),
                    "train/vbc": outputs["vbc_loss"].item(),
                    "progress": global_step / total_steps
                })
                pbar.set_description(f"Ep {epoch} Loss {loss.item():.4f}")

        # Save Checkpoint
        # save_path = f"./checkpoints/epoch_{epoch}"
        save_path = f"./checkpoints/{dataset_dir_name}/epoch_{epoch}"
        model_engine.save_checkpoint(save_path)
        
        if args.local_rank <= 0:
            hf_path = os.path.join(save_path, "hf_format")
            os.makedirs(hf_path, exist_ok=True)
            model_engine.module.base_model.save_pretrained(hf_path, safe_serialization=True)
            processor.save_pretrained(hf_path)
            torch.save(model_engine.module.projectors.state_dict(), os.path.join(hf_path, "projectors.bin"))

    if args.local_rank <= 0: wandb.finish()

if __name__ == "__main__":
    main()