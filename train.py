import os
import torch
import deepspeed
import argparse
import gc
from tqdm import tqdm
from datasets import load_dataset
from torch.utils.data import DataLoader, DistributedSampler
from transformers import Qwen2_5_VLProcessor

# 导入自定义模块
from configuration_latent import LatentConfig
from modeling_latent_qwen import LatentReasoningQwen
from dataset_latent import LatentReasoningDataset, collate_fn

# 环境配置
os.environ["TOKENIZERS_PARALLELISM"] = "false"

def parse_args():
    parser = argparse.ArgumentParser(description="Latent Reasoning Training (Sequence Expansion Mode)")
    parser.add_argument("--local_rank", type=int, default=-1, help="local rank for distributed training")
    parser.add_argument("--feature_dir", type=str, default="./data_preprocessed/aligned_features", help="预处理特征路径")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--save_path", type=str, default="./checkpoints/latent_qwen_expanded")
    parser = deepspeed.add_config_arguments(parser)
    return parser.parse_args()

def main():
    args = parse_args()

    # 1. 加载配置与处理器
    config = LatentConfig.load("config.yaml")
    processor = Qwen2_5_VLProcessor.from_pretrained(config.base_model)

    # 2. 注册并准备特殊 Token
    # 注意：这里的 Token 顺序和数量必须与建模代码中的逻辑严格对应
    special_tokens = [config.latent_start_token]
    for i in range(config.type1_count):
        special_tokens.append(f"{config.type1_base_token}{i}|>")
    
    num_added_tokens = processor.tokenizer.add_special_tokens({"additional_special_tokens": special_tokens})
    config.latent_start_id = processor.tokenizer.convert_tokens_to_ids(config.latent_start_token)
    
    print(f"Added {num_added_tokens} tokens. Latent Start ID: {config.latent_start_id}")

    # 3. 加载 ScienceQA 数据集并注入 ID (用于匹配预处理特征)
    raw_dataset = load_dataset("derek-thomas/ScienceQA", split="train")
    
    # 为每条数据添加索引，确保能找到对应的 .pt 特征文件
    def add_idx(example, idx):
        example.update({"id": f"train_{idx}"})
        return example
    indexed_dataset = raw_dataset.map(add_idx, with_indices=True)

    # 4. 初始化模型
    model = LatentReasoningQwen(config)
    
    # 必须 Resize Embedding 使得新 Token 生效
    model.base_model.resize_token_embeddings(len(processor.tokenizer))
    
    # 启用梯度检查点以节省显存 (序列扩展模式开销较大)
    model.base_model.gradient_checkpointing_enable()

    # 5. 准备 Dataset 和 DataLoader
    train_dataset = LatentReasoningDataset(
        data_list=indexed_dataset,
        processor=processor,
        config=config,
        feature_dir=args.feature_dir
    )

    sampler = DistributedSampler(train_dataset) if args.local_rank != -1 else None
    dataloader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        sampler=sampler,
        collate_fn=collate_fn,
        num_workers=4,
        pin_memory=True
    )

    # 6. DeepSpeed 初始化
    # 使用 zero2_config.json 中的优化策略
    model_engine, optimizer, _, _ = deepspeed.initialize(
        args=args,
        model=model,
        model_parameters=model.parameters(),
        config="zero2_config.json"
    )

    # 7. 训练循环
    model_engine.train()
    for epoch in range(args.epochs):
        if sampler:
            sampler.set_epoch(epoch)
        
        pbar = tqdm(dataloader, desc=f"Epoch {epoch}", disable=(args.local_rank != 0))
        
        for step, batch in enumerate(pbar):
            # 将数据搬运至设备
            input_ids = batch["input_ids"].to(model_engine.device)
            labels = batch["labels"].to(model_engine.device)
            pixel_values = batch["pixel_values"].to(model_engine.device, dtype=torch.bfloat16)
            image_grid_thw = batch["image_grid_thw"].to(model_engine.device)
            
            # 准备对齐特征
            alignment_features = {
                k: v.to(model_engine.device, dtype=torch.bfloat16)
                for k, v in batch["alignment_features"].items() if isinstance(v, torch.Tensor)
            }

            # 前向传播
            # 这里的输出是修改后的 modeling_latent_qwen 返回的字典
            outputs = model_engine(
                input_ids=input_ids,
                labels=labels,
                pixel_values=pixel_values,
                image_grid_thw=image_grid_thw,
                alignment_features=alignment_features
            )

            loss = outputs["loss"]
            sft_loss = outputs["sft_loss"]
            mse_loss = outputs["mse_loss"]
            
            # 反向传播
            model_engine.backward(loss)
            model_engine.step()

            # 日志监控
            if args.local_rank <= 0 and step % 5 == 0:
                pbar.set_postfix({
                    "Total": f"{loss.item():.3f}",
                    "SFT": f"{sft_loss.item():.3f}",
                    "MSE": f"{mse_loss.item():.4f}"
                })

        # 保存每个 Epoch 的 Checkpoint
        if args.local_rank <= 0:
            save_dir = os.path.join(args.save_path, f"epoch_{epoch}")
            model_engine.save_checkpoint(save_dir)
            print(f"Saved checkpoint to {save_dir}")

    print("Training Complete.")

if __name__ == "__main__":
    main()