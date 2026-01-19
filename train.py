import os
import torch
from torch_npu.contrib import transfer_to_npu
import torch.distributed as dist
import argparse
from tqdm import tqdm
from datasets import load_dataset
from torch.utils.data import DataLoader, DistributedSampler
from transformers import Qwen2_5_VLProcessor
import deepspeed
import wandb

from config.configuration_latent import LatentConfig
from model.modeling_latent_qwen import LatentReasoningQwen
from data.dataset_latent import LatentReasoningDataset, collate_fn

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--local_rank", type=int, default=int(os.environ.get("LOCAL_RANK", -1)))
    parser.add_argument("--feature_dir", type=str, required=True)
    # parser.add_argument("--epochs", type=int, default=3) # <--- 已移除，改用 config 读取
    
    # WandB 参数
    parser.add_argument("--wandb_project", type=str, default="latent-reasoning-vlm", help="WandB 项目名称")
    parser.add_argument("--wandb_run_name", type=str, default=None, help="WandB 运行名称")
    parser.add_argument("--wandb_entity", type=str, default=None, help="WandB 用户名")
    parser.add_argument("--wandb_offline", action="store_true", help="离线模式")
    
    parser = deepspeed.add_config_arguments(parser)
    return parser.parse_args()

def main():
    args = parse_args()
    
    if args.local_rank != -1:
        torch.cuda.set_device(args.local_rank)
        dist.init_process_group(backend="nccl")

    # 1. Load Config (包含 epochs)
    config = LatentConfig.load("./config/config.yaml")
    
    # === WandB Init ===
    if args.local_rank <= 0:
        wandb.init(
            project=args.wandb_project,
            name=args.wandb_run_name,
            entity=args.wandb_entity,
            mode="offline" if args.wandb_offline else "online",
            config={
                "model": config.base_model,
                "batch_size": config.batch_size,
                "epochs": config.epochs,  # <--- 使用 config.epochs
                "learning_rate": config.alpha_sft, # 注意：这里只是记录，实际LR由 DeepSpeed Config 控制
                "alpha_sft": config.alpha_sft,
                "beta_mse": config.beta_mse,
                "gradient_checkpointing": config.gradient_checkpointing,
                "stages": [s.name for s in config.stages]
            }
        )

    # 2. Tokenizer
    processor = Qwen2_5_VLProcessor.from_pretrained(config.base_model)
    new_tokens = [config.extract_token] + [stage.token for stage in config.stages]
    processor.tokenizer.add_special_tokens({"additional_special_tokens": new_tokens})
    
    config.extract_token_id = processor.tokenizer.convert_tokens_to_ids(config.extract_token)
    for stage in config.stages:
        stage.token_id = processor.tokenizer.convert_tokens_to_ids(stage.token)

    # 3. Model
    model = LatentReasoningQwen(config)
    model.base_model.resize_token_embeddings(len(processor.tokenizer))
    model.len_tokenizer = len(processor.tokenizer)

    # 4. Dataset
    raw_dataset = load_dataset("derek-thomas/ScienceQA", split="train")
    raw_dataset = raw_dataset.map(lambda x, i: {"id": f"train_{i}"}, with_indices=True)
    raw_dataset = raw_dataset.filter(lambda x: x['image'] is not None)
    
    train_dataset = LatentReasoningDataset(raw_dataset, processor, config, args.feature_dir)
    sampler = DistributedSampler(train_dataset) if args.local_rank != -1 else None
    dataloader = DataLoader(train_dataset, batch_size=config.batch_size, sampler=sampler, collate_fn=collate_fn)

    # 5. DeepSpeed Init
    model_engine, optimizer, _, _ = deepspeed.initialize(
        args=args, model=model, model_parameters=[p for p in model.parameters() if p.requires_grad]
    )

    # 使用 config.epochs 计算总步数
    total_steps = len(dataloader) * config.epochs 
    global_step = 0

    # 6. Training Loop (使用 config.epochs)
    for epoch in range(config.epochs):
        if sampler: sampler.set_epoch(epoch)
        pbar = tqdm(enumerate(dataloader), total=len(dataloader), disable=(args.local_rank > 0))
        
        for step, batch in pbar:
            global_step += 1
            
            batch_gpu = {}
            for k, v in batch.items():
                if isinstance(v, torch.Tensor):
                    batch_gpu[k] = v.to(model_engine.device)
                elif isinstance(v, dict):
                    batch_gpu[k] = {}
                    for sk, sv in v.items():
                        if isinstance(sv, torch.Tensor):
                            batch_gpu[k][sk] = sv.to(model_engine.device, dtype=torch.bfloat16)
                        else:
                            batch_gpu[k][sk] = sv
                else:
                    batch_gpu[k] = v

            outputs = model_engine(**batch_gpu)
            loss = outputs["loss"]
            
            model_engine.backward(loss)
            model_engine.step()
            
            if args.local_rank <= 0:
                current_lr = model_engine.optimizer.param_groups[0]['lr']
                progress_pct = (global_step / total_steps) * 100
                
                log_data = {
                    "train/total_loss": loss.item(),
                    "train/sft_loss": outputs["sft_loss"].item(),
                    "train/mse_loss": outputs["mse_loss"].item(),
                    "train/learning_rate": current_lr,
                    "train/epoch": epoch + (step + 1) / len(dataloader),
                    "train/global_step": global_step,
                    "train/progress_percentage": progress_pct
                }
                wandb.log(log_data)
                
                pbar.set_description(
                    f"Ep {epoch+1}/{config.epochs}| {progress_pct:.1f}% | "
                    f"Loss: {loss.item():.4f} (SFT:{outputs['sft_loss']:.3f} MSE:{outputs['mse_loss']:.3f})"
                )

        if args.local_rank <= 0:
            print("Saving checkpoint...")
            model_engine.save_checkpoint("./checkpoints/latent_qwen_final_{epoch}".format(epoch=epoch+1))
            wandb.finish()

if __name__ == "__main__":
    main()