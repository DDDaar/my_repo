import os
import torch
import deepspeed
import argparse
from tqdm import tqdm
from datasets import load_dataset
from torch.utils.data import DataLoader, DistributedSampler
from transformers import Qwen2_5_VLProcessor

# 导入自定义模块
from configuration_latent import LatentConfig
from modeling_latent_qwen import LatentReasoningQwen
from dataset_latent import LatentReasoningDataset, collate_fn

# 配置环境变量以优化 NPU/GPU 性能
os.environ["TOKENIZERS_PARALLELISM"] = "false"

def parse_args():
    parser = argparse.ArgumentParser(description="Latent Reasoning Training on ScienceQA")
    parser.add_argument("--local_rank", type=int, default=-1, help="local rank for distributed training")
    parser.add_argument("--feature_dir", type=str, default="./data_preprocessed/aligned_features", help="Path to pre-extracted .pt features")
    parser = deepspeed.add_config_arguments(parser)
    return parser.parse_args()

def main():
    args = parse_args()

    # 1. 加载配置与处理器
    config = LatentConfig.load("config.yaml")
    # Qwen2.5-VL 的处理器负责处理图像补丁（Patches）和文本分词
    processor = Qwen2_5_VLProcessor.from_pretrained(config.base_model)

    # 2. 注册并准备特殊 Token
    # 这里的逻辑是：给模型添加 <|latent_start|> 和 <|vision_extract_0|> 等 token
    special_tokens = [config.latent_start_token] + [
        f"{config.type1_base_token}{i}|>" for i in range(config.type1_count)
    ]
    num_added_tokens = processor.tokenizer.add_special_tokens({"additional_special_tokens": special_tokens})
    config.latent_start_id = processor.tokenizer.convert_tokens_to_ids(config.latent_start_token)

    # 3. 准备 ScienceQA 数据集并确保 ID 对齐
    print(f"Loading ScienceQA and aligning IDs...")
    raw_dataset = load_dataset("derek-thomas/ScienceQA", split="train")
    
    # 【核心对齐逻辑】：先通过 map 赋予物理索引 ID，再过滤有图片的样本
    # 这确保了 Dataset[idx] 获取到的 id 与预处理保存的 {id}.pt 文件名完全一致
    indexed_dataset = raw_dataset.map(lambda x, idx: {'id': f"train_{idx}"}, with_indices=True)
    train_data = indexed_dataset.filter(lambda x: x['image'] is not None)

    train_dataset = LatentReasoningDataset(
        train_data, 
        processor, 
        config, 
        args.feature_dir
    )

    # 4. 初始化模型
    # 模型会将 Qwen2.5-VL 封装在内，并在 forward 中执行隐藏层的“思考”循环
    model = LatentReasoningQwen(config)
    # 必须根据新添加的特殊 token 调整 Embedding 层大小
    model.base_model.resize_token_embeddings(len(processor.tokenizer))

    # 5. DeepSpeed 初始化
    # 自动管理 ZeRO 优化、混合精度 (BF16) 和 Adam 优化器
    model_engine, optimizer, _, _ = deepspeed.initialize(
        args=args,
        model=model,
        model_parameters=model.parameters(),
        # 需确保目录下有 ds_config.json
        config="ds_config.json" 
    )

    # 6. 数据加载器 (支持多卡分布式)
    sampler = DistributedSampler(train_dataset)
    dataloader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        collate_fn=collate_fn,
        sampler=sampler,
        num_workers=4,
        pin_memory=True
    )

    # 7. 训练循环
    model_engine.train()
    epochs = 5
    for epoch in range(epochs):
        sampler.set_epoch(epoch)
        pbar = tqdm(dataloader, desc=f"Epoch {epoch}", disable=(args.local_rank != 0))
        
        for step, batch in enumerate(pbar):
            # 将数据搬运至 NPU/GPU
            input_ids = batch["input_ids"].to(model_engine.device)
            labels = batch["labels"].to(model_engine.device)
            pixel_values = batch["pixel_values"].to(model_engine.device, dtype=torch.bfloat16)
            image_grid_thw = batch["image_grid_thw"].to(model_engine.device)
            
            # 这里的对齐特征是预处理好的 DINO/Depth/Seg 特征
            alignment_features = {
                k: v.to(model_engine.device, dtype=torch.bfloat16)
                for k, v in batch["alignment_features"].items() if isinstance(v, torch.Tensor)
            }

            # 前向传播：返回组合 Loss (SFT Loss + MSE Alignment Loss)
            outputs = model_engine(
                input_ids=input_ids,
                labels=labels,
                pixel_values=pixel_values,
                image_grid_thw=image_grid_thw,
                alignment_features=alignment_features
            )

            loss = outputs["loss"]
            
            # 反向传播与优化
            model_engine.backward(loss)
            model_engine.step()

            if args.local_rank == 0:
                pbar.set_postfix({"loss": loss.item()})

        # 每个 Epoch 保存一次权重
        if args.local_rank == 0:
            model_engine.save_checkpoint(f"checkpoints/epoch_{epoch}")

if __name__ == "__main__":
    main()