import os
import torch
import torch.distributed as dist
import argparse
from tqdm import tqdm
from datasets import load_dataset
from torch.utils.data import DataLoader, DistributedSampler
from transformers import Qwen2_5_VLProcessor

try:
    import torch_npu
    from torch_npu.contrib import transfer_to_npu
except ImportError:
    pass

import deepspeed
from config.configuration_latent import LatentConfig
from model.modeling_latent_qwen import LatentReasoningQwen
from data.dataset_latent import LatentReasoningDataset, collate_fn

def parse_args():
    parser = argparse.ArgumentParser(description="Latent Reasoning Training")
    parser.add_argument("--local_rank", type=int, default=int(os.environ.get("LOCAL_RANK", -1)))
    parser.add_argument("--feature_dir", type=str, required=True)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--save_path", type=str, default="./checkpoints/latent_qwen")
    parser = deepspeed.add_config_arguments(parser)
    return parser.parse_args()

def main():
    args = parse_args()

    if args.local_rank != -1:
        if hasattr(torch, "npu") and torch.npu.is_available():
            backend = "hccl"
            torch.npu.set_device(args.local_rank)
        else:
            backend = "nccl"
            torch.cuda.set_device(args.local_rank)
        dist.init_process_group(backend=backend)

    config = LatentConfig.load("./config/config.yaml")
    processor = Qwen2_5_VLProcessor.from_pretrained(config.base_model)

    special_tokens = [config.latent_start_token]
    for i in range(config.type1_count):
        special_tokens.append(f"{config.type1_base_token}{i}|>")
    processor.tokenizer.add_special_tokens({"additional_special_tokens": special_tokens})
    config.latent_start_id = processor.tokenizer.convert_tokens_to_ids(config.latent_start_token)

    raw_dataset = load_dataset("derek-thomas/ScienceQA", split="train")
    indexed_dataset = raw_dataset.map(lambda x, i: {"id": f"train_{i}"}, with_indices=True)
    indexed_dataset = indexed_dataset.filter(lambda x: x['image'] is not None)

    model = LatentReasoningQwen(config)
    model.base_model.resize_token_embeddings(len(processor.tokenizer))
    
    train_dataset = LatentReasoningDataset(indexed_dataset, processor, config, args.feature_dir)
    sampler = DistributedSampler(train_dataset) if args.local_rank != -1 else None
    dataloader = DataLoader(train_dataset, batch_size=config.batch_size, sampler=sampler, collate_fn=collate_fn)

    model_engine, optimizer, _, _ = deepspeed.initialize(
        args=args, model=model, model_parameters=[p for p in model.parameters() if p.requires_grad]
    )

    for epoch in range(args.epochs):
        if sampler: sampler.set_epoch(epoch)
        pbar = tqdm(dataloader, disable=(args.local_rank > 0))
        for batch in pbar:
            # 搬运数据，注意 image_grid_thw 必须是 long 类型
            input_ids = batch["input_ids"].to(model_engine.device)
            labels = batch["labels"].to(model_engine.device)
            pixel_values = batch["pixel_values"].to(model_engine.device, dtype=torch.bfloat16)
            image_grid_thw = batch["image_grid_thw"].to(model_engine.device, dtype=torch.long)
            
            alignment_features = {k: v.to(model_engine.device, dtype=torch.bfloat16) 
                                 for k, v in batch["alignment_features"].items() if isinstance(v, torch.Tensor)}

            outputs = model_engine(input_ids, labels, pixel_values, image_grid_thw, alignment_features)
            model_engine.backward(outputs["loss"])
            model_engine.step()

if __name__ == "__main__":
    main()