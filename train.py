# import os
# import torch
# from torch_npu.contrib import transfer_to_npu # 若非 Ascend NPU 可注释
# import torch.distributed as dist
# import argparse
# import random
# import numpy as np
# from tqdm import tqdm
# from torch.utils.data import DataLoader, DistributedSampler
# from transformers import Qwen2_5_VLProcessor
# import deepspeed
# import wandb

# from config.configuration_latent import LatentConfig
# from model.modeling_latent_qwen import LatentReasoningQwen
# from data.dataset_latent import LatentReasoningDataset, collate_fn

# def set_seed(seed=42):
#     random.seed(seed)
#     np.random.seed(seed)
#     torch.manual_seed(seed)
#     torch.cuda.manual_seed_all(seed)
#     torch.backends.cudnn.deterministic = True
#     torch.backends.cudnn.benchmark = False

# def seed_worker(worker_id):
#     worker_seed = torch.initial_seed() % 2**32
#     np.random.seed(worker_seed)
#     random.seed(worker_seed)

# def parse_args():
#     parser = argparse.ArgumentParser()
#     parser.add_argument("--local_rank", type=int, default=int(os.environ.get("LOCAL_RANK", -1)))
#     parser.add_argument("--wandb_project", type=str, default="latent_vbc_full")
#     parser.add_argument("--wandb_run_name", type=str, default="run_standardized")
#     parser.add_argument("--wandb_offline", action="store_true")
#     parser = deepspeed.add_config_arguments(parser)
#     return parser.parse_args()

# def main():
#     args = parse_args()

#     # 分布式初始化
#     if args.local_rank != -1:
#         torch.cuda.set_device(args.local_rank)
#         dist.init_process_group(backend="hccl") # 或 nccl

#     # 加载配置
#     config = LatentConfig.load("./config/config.yaml")

#     if len(config.train_datasets) == 1:
#         dataset_dir_name = config.train_datasets[0].name.replace("/", "_")
#     else:
#         dataset_dir_name = "mixed_datasets"
    
#     set_seed(config.seed) 

#     if args.local_rank <= 0:
#         print(f"--- Training Configuration ---")
#         print(f"Seed: {config.seed}")
#         print(f"Extract Prefix: '{config.extract_text_prefix}'")
#         print(f"VBC Enabled: Lambda={config.lambda_vbc}, Margin={config.vbc_margin}")
#         print(f"Target Scope: {config.vbc_target_scope}")
        
#         wandb.init(
#             project=args.wandb_project,
#             name=args.wandb_run_name,
#             mode="offline" if args.wandb_offline else "online",
#             config=config.__dict__
#         )

#     # === [关键步骤 1] 初始化 Processor 并注册 Token ===
#     # 必须在 Dataset 初始化之前完成，否则 Config 里没有正确的 Token ID
#     processor = Qwen2_5_VLProcessor.from_pretrained(config.base_model)
    
#     new_tokens = config.get_all_special_tokens()
#     processor.tokenizer.add_special_tokens({"additional_special_tokens": new_tokens})

#     # === [关键步骤 2] 将 Token ID 回填到 Config ===
#     config.extract_token_id = processor.tokenizer.convert_tokens_to_ids(config.extract_token)
#     config.answer_start_id = processor.tokenizer.convert_tokens_to_ids(config.answer_start)
#     config.answer_end_id = processor.tokenizer.convert_tokens_to_ids(config.answer_end)
    
#     for stage in config.stages:
#         stage.token_id = processor.tokenizer.convert_tokens_to_ids(stage.token)

#     # === [关键步骤 3] 初始化 Dataset (此时 Config 已包含 ID) ===
#     train_dataset = LatentReasoningDataset(processor, config)
    
#     # Model
#     model = LatentReasoningQwen(config)
#     # Resize embedding 必须包含所有新 Token
#     model.base_model.resize_token_embeddings(len(processor.tokenizer))
#     model.len_tokenizer = len(processor.tokenizer)

#     # Sampler & Loader
#     sampler = DistributedSampler(train_dataset) if args.local_rank != -1 else None
    
#     g = torch.Generator()
#     g.manual_seed(config.seed)

#     dataloader = DataLoader(
#         train_dataset,
#         batch_size=config.batch_size,
#         sampler=sampler,
#         collate_fn=collate_fn,
#         num_workers=4,
#         pin_memory=True,
#         worker_init_fn=seed_worker,
#         generator=g
#     )

#     # DeepSpeed Init
#     model_engine, optimizer, _, _ = deepspeed.initialize(
#         args=args,
#         model=model,
#         model_parameters=[p for p in model.parameters() if p.requires_grad]
#     )

#     # Training Loop
#     global_step = 0
#     total_steps = len(dataloader) * config.epochs
    
#     for epoch in range(config.epochs):
#         if sampler: sampler.set_epoch(epoch)
        
#         pbar = tqdm(enumerate(dataloader), total=len(dataloader), disable=(args.local_rank > 0))
        
#         for step, batch in pbar:
#             global_step += 1
            
#             # Move to device
#             batch_gpu = {}
#             for k, v in batch.items():
#                 if isinstance(v, torch.Tensor):
#                     batch_gpu[k] = v.to(model_engine.device)
#                 elif isinstance(v, dict): 
#                     batch_gpu[k] = {sk: sv.to(model_engine.device, dtype=torch.bfloat16) for sk, sv in v.items()}
#                 else:
#                     batch_gpu[k] = v

#             outputs = model_engine(**batch_gpu)
#             loss = outputs["loss"]
            
#             model_engine.backward(loss)
#             model_engine.step()

#             if args.local_rank <= 0:
#                 wandb.log({
#                     "train/loss": loss.item(),
#                     "train/sft_ans": outputs["sft_loss"].item(),
#                     "train/latent_ce": outputs["latent_ce"].item(),
#                     "train/mse": outputs["mse_loss"].item(),
#                     "train/vbc": outputs["vbc_loss"].item(),
#                     "progress": global_step / total_steps
#                 })
#                 pbar.set_description(f"Ep {epoch} Loss {loss.item():.4f}")

#         # Save Checkpoint
#         save_path = f"./checkpoints/{dataset_dir_name}/epoch_{epoch}"
#         model_engine.save_checkpoint(save_path)
        
#         if args.local_rank <= 0:
#             hf_path = os.path.join(save_path, "hf_format")
#             os.makedirs(hf_path, exist_ok=True)
#             model_engine.module.base_model.save_pretrained(hf_path, safe_serialization=True)
#             processor.save_pretrained(hf_path)
#             torch.save(model_engine.module.projectors.state_dict(), os.path.join(hf_path, "projectors.bin"))

#     if args.local_rank <= 0: wandb.finish()

# if __name__ == "__main__":
#     main()




import os
import torch
from torch_npu.contrib import transfer_to_npu # 若非 Ascend NPU 可注释
import torch.distributed as dist
import argparse
import random
import copy
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
    parser.add_argument("--wandb_run_name", type=str, default="run_standardized")
    parser.add_argument("--wandb_offline", action="store_true")
    
    # 新增：控制验证频率和采样数量
    parser.add_argument("--eval_steps", type=int, default=2000, help="每隔多少步进行一次验证")
    parser.add_argument("--eval_max_batches", type=int, default=5, help="每次验证最多跑多少个 batch")

    parser = deepspeed.add_config_arguments(parser)
    return parser.parse_args()

def evaluate_loss(model_engine, dataloader, device, max_batches=None):
    """
    计算验证集上的平均 Loss。
    支持 max_batches 参数，仅验证部分数据以节省时间。
    """
    model_engine.eval()
    total_loss = 0.0
    total_steps = 0
    
    desc = f"Validating ({'Partial' if max_batches else 'Full'})"
    
    # 禁用 tqdm 在非主进程上的输出
    disable_tqdm = (dist.is_initialized() and dist.get_rank() > 0)
    
    with torch.no_grad():
        for i, batch in tqdm(enumerate(dataloader), desc=desc, leave=False, disable=disable_tqdm, total=max_batches if max_batches else len(dataloader)):
            if max_batches is not None and i >= max_batches:
                break

            # 移动数据到设备
            batch_gpu = {}
            for k, v in batch.items():
                if isinstance(v, torch.Tensor):
                    batch_gpu[k] = v.to(device)
                elif isinstance(v, dict): 
                    batch_gpu[k] = {sk: sv.to(device, dtype=torch.bfloat16) for sk, sv in v.items()}
                else:
                    batch_gpu[k] = v

            outputs = model_engine(**batch_gpu)
            loss = outputs["loss"]
            
            total_loss += loss.item()
            total_steps += 1

    if total_steps == 0:
        model_engine.train()
        return 0.0

    metrics = torch.tensor([total_loss, total_steps], device=device)
    if dist.is_initialized():
        dist.all_reduce(metrics, op=dist.ReduceOp.SUM)
    
    avg_loss = metrics[0] / (metrics[1] + 1e-8)
    
    model_engine.train()
    return avg_loss.item()

def main():
    args = parse_args()

    if args.local_rank != -1:
        torch.cuda.set_device(args.local_rank)
        dist.init_process_group(backend="hccl") # 或 nccl

    config = LatentConfig.load("./config/config.yaml")

    if len(config.train_datasets) == 1:
        dataset_dir_name = config.train_datasets[0].name.replace("/", "_")
    else:
        dataset_dir_name = "mixed_datasets"
    
    set_seed(config.seed) 

    if args.local_rank <= 0:
        print(f"--- Training Configuration ---")
        print(f"Seed: {config.seed}")
        print(f"VBC Enabled: Lambda={config.lambda_vbc}")
        
        wandb.init(
            project=args.wandb_project,
            name=args.wandb_run_name,
            mode="offline" if args.wandb_offline else "online",
            config=config.__dict__
        )

    # === 1. 初始化 Processor ===
    processor = Qwen2_5_VLProcessor.from_pretrained(config.base_model)
    new_tokens = config.get_all_special_tokens()
    processor.tokenizer.add_special_tokens({"additional_special_tokens": new_tokens})

    # === 2. 回填 Token ID ===
    config.extract_token_id = processor.tokenizer.convert_tokens_to_ids(config.extract_token)
    config.answer_start_id = processor.tokenizer.convert_tokens_to_ids(config.answer_start)
    config.answer_end_id = processor.tokenizer.convert_tokens_to_ids(config.answer_end)
    for stage in config.stages:
        stage.token_id = processor.tokenizer.convert_tokens_to_ids(stage.token)

    # === 3. 训练数据集 ===
    train_dataset = LatentReasoningDataset(processor, config)
    
    # === 4. 验证数据集 (修正 Feature Dir 逻辑) ===
    val_loader = None
    if len(config.train_datasets) == 1:
        if args.local_rank <= 0:
            print("[Info] 检测到 Single 模式，尝试加载验证集...")
        
        splits_to_try = ['validation', 'test']
        val_dataset = None
        
        for split_name in splits_to_try:
            val_config = copy.deepcopy(config)
            target_ds_cfg = val_config.train_datasets[0]
            
            # 1. 修改 Split
            target_ds_cfg.split = split_name
            
            # 2. 修正 Feature Dir
            # 逻辑：如果原路径包含 "train"，尝试替换为当前的 split (如 "test")
            # 假设路径格式类似: .../aligned_features_ScienceQA_train
            original_feat_dir = target_ds_cfg.feature_dir
            if original_feat_dir and "train" in original_feat_dir:
                new_feat_dir = original_feat_dir.replace("train", split_name)
                
                # 检查新路径是否存在
                if os.path.exists(new_feat_dir):
                    target_ds_cfg.feature_dir = new_feat_dir
                    if args.local_rank <= 0:
                        print(f" -> [Auto-Fix] 将验证集 feature_dir 修正为: {new_feat_dir}")
                else:
                    # 如果找不到对应的验证集特征目录，设为空，防止加载时因找不到文件而过滤所有数据
                    if args.local_rank <= 0:
                        print(f" -> [Warning] 找不到目录 {new_feat_dir}。验证集将不加载对齐特征(MSE无效)，但保留数据用于计算 Loss。")
                    target_ds_cfg.feature_dir = ""
            
            try:
                temp_ds = LatentReasoningDataset(processor, val_config)
                if len(temp_ds) > 0:
                    val_dataset = temp_ds
                    if args.local_rank <= 0:
                        print(f" -> 成功加载 '{split_name}' 集，共 {len(val_dataset)} 条样本。")
                    break
            except Exception as e:
                if args.local_rank <= 0:
                    print(f" -> 加载 '{split_name}' 失败: {e}")
                continue
        
        if val_dataset:
            val_sampler = DistributedSampler(val_dataset, shuffle=False) if args.local_rank != -1 else None
            val_loader = DataLoader(
                val_dataset,
                batch_size=config.batch_size,
                sampler=val_sampler,
                collate_fn=collate_fn,
                num_workers=4,
                pin_memory=True
            )

    # Model
    model = LatentReasoningQwen(config)
    model.base_model.resize_token_embeddings(len(processor.tokenizer))
    model.len_tokenizer = len(processor.tokenizer)

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

    model_engine, optimizer, _, _ = deepspeed.initialize(
        args=args,
        model=model,
        model_parameters=[p for p in model.parameters() if p.requires_grad]
    )

    global_step = 0
    
    for epoch in range(config.epochs):
        if sampler: sampler.set_epoch(epoch)
        
        model_engine.train()
        pbar = tqdm(enumerate(dataloader), total=len(dataloader), disable=(args.local_rank > 0))
        
        for step, batch in pbar:
            global_step += 1
            
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
                    "progress": global_step / (len(dataloader) * config.epochs),
                    "epoch": epoch
                }, step=global_step)
                pbar.set_description(f"Ep {epoch} Loss {loss.item():.4f}")

            # === Iteration-based Validation ===
            if val_loader is not None and global_step % args.eval_steps == 0:
                if args.local_rank <= 0:
                    print(f"\n[Step {global_step}] Running Partial Validation...")
                
                val_loss = evaluate_loss(model_engine, val_loader, model_engine.device, max_batches=args.eval_max_batches)
                
                if args.local_rank <= 0:
                    print(f"[Step {global_step}] Val Loss: {val_loss:.4f}")
                    wandb.log({
                        "val/loss": val_loss,
                        "val/step": global_step
                    }, step=global_step)

        # Save Checkpoint
        save_path = f"./checkpoints/{dataset_dir_name}/epoch_{epoch}"
        model_engine.save_checkpoint(save_path)
        
        if args.local_rank <= 0:
            hf_path = os.path.join(save_path, "hf_format")
            os.makedirs(hf_path, exist_ok=True)
            model_engine.module.base_model.save_pretrained(hf_path, safe_serialization=True)
            processor.save_pretrained(hf_path)
            torch.save(model_engine.module.projectors.state_dict(), os.path.join(hf_path, "projectors.bin"))
            print('hf格式模型保存完毕！')

    if args.local_rank <= 0: wandb.finish()

if __name__ == "__main__":
    main()