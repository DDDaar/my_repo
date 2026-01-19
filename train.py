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

    # === WandB 参数（运行控制，不是训练超参） ===
    parser.add_argument("--wandb_project", type=str, default="my_repo")
    parser.add_argument("--wandb_run_name", type=str, default="first_success")
    parser.add_argument("--wandb_entity", type=str, default=None)
    parser.add_argument("--wandb_offline", action="store_true")

    parser = deepspeed.add_config_arguments(parser)
    return parser.parse_args()


def main():
    args = parse_args()

    # === 分布式初始化 ===
    if args.local_rank != -1:
        torch.cuda.set_device(args.local_rank)
        dist.init_process_group(backend="nccl")

    # === 1. Load YAML Config ===
    config = LatentConfig.load("./config/config.yaml")
    epochs = config.epochs   # ✅ 唯一的 epochs 来源

    # === WandB 初始化（仅 rank0） ===
    if args.local_rank <= 0:
        wandb.init(
            project=args.wandb_project,
            name=args.wandb_run_name,
            entity=args.wandb_entity,
            mode="offline" if args.wandb_offline else "online",
            config={
                "model": config.base_model,
                "batch_size": config.batch_size,
                "learning_rate": config.alpha_sft,
                "alpha_sft": config.alpha_sft,
                "beta_mse": config.beta_mse,
                "epochs": epochs,
                "gradient_checkpointing": config.gradient_checkpointing,
                "stages": [s.name for s in config.stages]
            }
        )

    # === 2. Processor / Tokenizer ===
    processor = Qwen2_5_VLProcessor.from_pretrained(config.base_model)

    new_tokens = [config.extract_token] + [stage.token for stage in config.stages]
    processor.tokenizer.add_special_tokens({"additional_special_tokens": new_tokens})

    config.extract_token_id = processor.tokenizer.convert_tokens_to_ids(config.extract_token)
    for stage in config.stages:
        stage.token_id = processor.tokenizer.convert_tokens_to_ids(stage.token)

    # === 3. Model ===
    model = LatentReasoningQwen(config)

    model.base_model.resize_token_embeddings(len(processor.tokenizer))
    model.len_tokenizer = len(processor.tokenizer)

    # === 4. Dataset ===
    raw_dataset = load_dataset("derek-thomas/ScienceQA", split="train")
    raw_dataset = raw_dataset.map(lambda x, i: {"id": f"train_{i}"}, with_indices=True)
    raw_dataset = raw_dataset.filter(lambda x: x["image"] is not None)

    train_dataset = LatentReasoningDataset(
        raw_dataset, processor, config, args.feature_dir
    )

    sampler = DistributedSampler(train_dataset) if args.local_rank != -1 else None
    dataloader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        sampler=sampler,
        collate_fn=collate_fn
    )

    # === 5. DeepSpeed Init ===
    model_engine, optimizer, _, _ = deepspeed.initialize(
        args=args,
        model=model,
        model_parameters=[p for p in model.parameters() if p.requires_grad]
    )

    total_steps = len(dataloader) * epochs
    global_step = 0

    # === 6. Training Loop ===
    for epoch in range(epochs):
        if sampler:
            sampler.set_epoch(epoch)

        pbar = tqdm(
            enumerate(dataloader),
            total=len(dataloader),
            disable=(args.local_rank > 0)
        )

        for step, batch in pbar:
            global_step += 1

            batch_gpu = {}
            for k, v in batch.items():
                if isinstance(v, torch.Tensor):
                    batch_gpu[k] = v.to(model_engine.device)
                elif isinstance(v, dict):
                    batch_gpu[k] = {
                        sk: sv.to(model_engine.device, dtype=torch.bfloat16)
                        if isinstance(sv, torch.Tensor) else sv
                        for sk, sv in v.items()
                    }
                else:
                    batch_gpu[k] = v

            outputs = model_engine(**batch_gpu)
            loss = outputs["loss"]

            model_engine.backward(loss)
            model_engine.step()

            if args.local_rank <= 0:
                lr = model_engine.optimizer.param_groups[0]["lr"]
                progress_pct = global_step / total_steps * 100.0

                wandb.log({
                    "train/total_loss": loss.item(),
                    "train/sft_loss": outputs["sft_loss"].item(),
                    "train/mse_loss": outputs["mse_loss"].item(),
                    "train/learning_rate": lr,
                    "train/epoch": epoch + (step + 1) / len(dataloader),
                    "train/global_step": global_step,
                    "train/progress_percentage": progress_pct
                })

                pbar.set_description(
                    f"Ep {epoch} | {progress_pct:.1f}% | "
                    f"Loss {loss.item():.4f} "
                    f"(SFT {outputs['sft_loss']:.3f} / MSE {outputs['mse_loss']:.3f})"
                )

        # === Save Checkpoint (rank0 only) ===
        if args.local_rank <= 0:
            save_dir = f"./checkpoints/latent_qwen_epoch_{epoch}训练后"
            print(f"Saving checkpoint to {save_dir}")
            model_engine.save_checkpoint(save_dir)

    if args.local_rank <= 0:
        wandb.finish()


if __name__ == "__main__":
    main()
