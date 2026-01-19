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

from config.configuration_latent import LatentConfig
from model.modeling_latent_qwen import LatentReasoningQwen
from data.dataset_latent import LatentReasoningDataset, collate_fn

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--local_rank", type=int, default=int(os.environ.get("LOCAL_RANK", -1)))
    parser.add_argument("--feature_dir", type=str, required=True)
    parser.add_argument("--epochs", type=int, default=3)
    parser = deepspeed.add_config_arguments(parser)
    return parser.parse_args()

def main():
    args = parse_args()
    if args.local_rank != -1:
        torch.cuda.set_device(args.local_rank)
        dist.init_process_group(backend="nccl")

    # 1. Load Config
    config = LatentConfig.load("./config/config.yaml")
    
    # 2. Tokenizer
    processor = Qwen2_5_VLProcessor.from_pretrained(config.base_model)
    
    # 收集并添加特殊 Tokens
    new_tokens = [config.extract_token] + [stage.token for stage in config.stages]

    print(f'添加special tokens前，len(tokenizer)={len(processor.tokenizer)}')

    processor.tokenizer.add_special_tokens({"additional_special_tokens": new_tokens})
    
    config.extract_token_id = processor.tokenizer.convert_tokens_to_ids(config.extract_token)
    for stage in config.stages:
        stage.token_id = processor.tokenizer.convert_tokens_to_ids(stage.token)

    # 3. Model
    model = LatentReasoningQwen(config)

    print(f'添加special tokens后，resize前，len(tokenizer)={len(processor.tokenizer)}')


    model.base_model.resize_token_embeddings(len(processor.tokenizer))

    print(f'添加special tokens后，len(tokenizer)={len(processor.tokenizer)}')
    len_tokenizer = len(processor.tokenizer)
    model.len_tokenizer = len_tokenizer
    
    # 4. Dataset
    raw_dataset = load_dataset("derek-thomas/ScienceQA", split="train")
    raw_dataset = raw_dataset.map(lambda x, i: {"id": f"train_{i}"}, with_indices=True)
    raw_dataset = raw_dataset.filter(lambda x: x['image'] is not None)
    
    train_dataset = LatentReasoningDataset(raw_dataset, processor, config, args.feature_dir)
    sampler = DistributedSampler(train_dataset) if args.local_rank != -1 else None
    dataloader = DataLoader(train_dataset, batch_size=config.batch_size, sampler=sampler, collate_fn=collate_fn)

    # 5. DeepSpeed
    model_engine, optimizer, _, _ = deepspeed.initialize(
        args=args, model=model, model_parameters=[p for p in model.parameters() if p.requires_grad]
    )

    # 6. Training Loop
    for epoch in range(args.epochs):
        if sampler: sampler.set_epoch(epoch)
        pbar = tqdm(dataloader, disable=(args.local_rank > 0))
        
        for batch in pbar:
            # 安全移动数据到 GPU/NPU
            batch_gpu = {}
            for k, v in batch.items():
                if isinstance(v, torch.Tensor):
                    batch_gpu[k] = v.to(model_engine.device)
                elif isinstance(v, dict): # alignment_features
                    batch_gpu[k] = {}
                    for sk, sv in v.items():
                        if isinstance(sv, torch.Tensor):
                            batch_gpu[k][sk] = sv.to(model_engine.device, dtype=torch.bfloat16)
                        else:
                            batch_gpu[k][sk] = sv # 保持 ID 字符串原样
                else:
                    batch_gpu[k] = v

            outputs = model_engine(**batch_gpu)
            model_engine.backward(outputs["loss"])
            model_engine.step()
            
            if args.local_rank <= 0:
                pbar.set_description(f"Loss: {outputs['loss'].item():.4f} (SFT: {outputs['sft_loss']:.4f}, MSE: {outputs['mse_loss']:.4f})")

    if args.local_rank <= 0:
        model_engine.save_checkpoint("./checkpoints/latent_qwen_blind")

if __name__ == "__main__":
    main()