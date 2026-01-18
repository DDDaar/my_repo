import os
import torch
import torch.distributed as dist
import argparse
from tqdm import tqdm
from datasets import load_dataset
from torch.utils.data import DataLoader, DistributedSampler
from transformers import Qwen2_5_VLProcessor

# 昇腾 NPU 支持
try:
    import torch_npu
    from torch_npu.contrib import transfer_to_npu
except ImportError:
    pass

import deepspeed

# 假设这些是你的本地模块
from config.configuration_latent import LatentConfig
from model.modeling_latent_qwen import LatentReasoningQwen
from data.dataset_latent import LatentReasoningDataset, collate_fn

os.environ["TOKENIZERS_PARALLELISM"] = "false"

def parse_args():
    parser = argparse.ArgumentParser(description="Latent Reasoning Training")
    # 增加对环境变量的读取，解决 DeepSpeed 启动报错问题
    parser.add_argument("--local_rank", type=int, default=int(os.environ.get("LOCAL_RANK", -1)), 
                        help="local rank for distributed training")
    parser.add_argument("--feature_dir", type=str, required=True)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--save_path", type=str, default="./checkpoints/latent_qwen")
    parser = deepspeed.add_config_arguments(parser)
    return parser.parse_args()

def main():
    args = parse_args()

    # --- 核心修改 1: 鲁棒的分布式初始化 ---
    if args.local_rank != -1:
        if torch.cuda.is_available():
            backend = "nccl"
            device_str = f"cuda:{args.local_rank}"
            torch.cuda.set_device(args.local_rank)
        elif hasattr(torch, "npu") and torch.npu.is_available():
            backend = "hccl"
            device_str = f"npu:{args.local_rank}"
            torch.npu.set_device(args.local_rank)
        else:
            backend = "gloo"
            device_str = "cpu"
        
        dist.init_process_group(backend=backend)
        print(f"Initialized Process Group: rank {args.local_rank}, backend {backend}")

    # 加载配置和处理器
    config = LatentConfig.load("./config/config.yaml")
    processor = Qwen2_5_VLProcessor.from_pretrained(config.base_model)

    # 注册特殊 Token
    special_tokens = [config.latent_start_token]
    for i in range(config.type1_count):
        special_tokens.append(f"{config.type1_base_token}{i}|>")
    
    processor.tokenizer.add_special_tokens({"additional_special_tokens": special_tokens})
    config.latent_start_id = processor.tokenizer.convert_tokens_to_ids(config.latent_start_token)

    # 加载数据
    raw_dataset = load_dataset("derek-thomas/ScienceQA", split="train")
    def add_idx(example, idx):
        example.update({"id": f"train_{idx}"})
        return example
    indexed_dataset = raw_dataset.map(add_idx, with_indices=True)

    # 初始化模型
    model = LatentReasoningQwen(config)
    model.base_model.resize_token_embeddings(len(processor.tokenizer))
    
    # 显存优化配置
    if hasattr(model.base_model, "gradient_checkpointing_enable"):
        model.base_model.gradient_checkpointing_enable()
    # 针对某些模型架构的特殊处理
    if hasattr(model.base_model, "model") and hasattr(model.base_model.model, "embed_tokens"):
        model.base_model.model.embed_tokens.gradient_checkpointing = True

    train_dataset = LatentReasoningDataset(
        data_list=indexed_dataset,
        processor=processor,
        config=config,
        feature_dir=args.feature_dir
    )

    # --- 核心修改 2: 只有在初始化后才能调用 DistributedSampler ---
    sampler = DistributedSampler(train_dataset) if args.local_rank != -1 else None
    dataloader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        sampler=sampler,
        collate_fn=collate_fn,
        num_workers=2, 
        pin_memory=True
    )

    # DeepSpeed 初始化
    # model_parameters 仅包含需要梯度的参数
    model_engine, optimizer, _, _ = deepspeed.initialize(
        args=args,
        model=model,
        model_parameters=[p for p in model.parameters() if p.requires_grad]
    )

    model_engine.train()
    for epoch in range(args.epochs):
        if sampler:
            sampler.set_epoch(epoch)
        
        pbar = tqdm(dataloader, desc=f"Epoch {epoch}", disable=(args.local_rank > 0))
        for step, batch in enumerate(pbar):
            # 数据搬运：使用 model_engine.device 自动适配 NPU/GPU
            input_ids = batch["input_ids"].to(model_engine.device)
            labels = batch["labels"].to(model_engine.device)
            pixel_values = batch["pixel_values"].to(model_engine.device, dtype=torch.bfloat16)
            image_grid_thw = batch["image_grid_thw"].to(model_engine.device)
            
            alignment_features = {
                k: v.to(model_engine.device, dtype=torch.bfloat16)
                for k, v in batch["alignment_features"].items() if isinstance(v, torch.Tensor)
            }

            outputs = model_engine(
                input_ids=input_ids,
                labels=labels,
                pixel_values=pixel_values,
                image_grid_thw=image_grid_thw,
                alignment_features=alignment_features
            )

            loss = outputs["loss"]
            model_engine.backward(loss)
            model_engine.step()

            if args.local_rank <= 0 and step % 5 == 0:
                pbar.set_postfix({"Loss": f"{loss.item():.4f}"})

        # 保存 Checkpoint
        if args.local_rank <= 0:
            # DeepSpeed 推荐的保存方式
            model_engine.save_checkpoint(args.save_path, tag=f"epoch_{epoch}")

if __name__ == "__main__":
    main()