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
    
    # 控制验证频率和采样数量
    parser.add_argument("--eval_steps", type=int, default=2000, help="每隔多少步进行一次验证")
    parser.add_argument("--eval_max_batches", type=int, default=5, help="每次验证最多跑多少个 batch")

    parser = deepspeed.add_config_arguments(parser)
    return parser.parse_args()

def evaluate_loss(model_engine, dataloader, device, max_batches=None):
    """
    计算验证集上的平均 Loss。
    """
    model_engine.eval()
    total_loss = 0.0
    total_steps = 0
    
    desc = f"Validating ({'Partial' if max_batches else 'Full'})"
    
    disable_tqdm = (dist.is_initialized() and dist.get_rank() > 0)
    
    with torch.no_grad():
        for i, batch in tqdm(enumerate(dataloader), desc=desc, leave=False, disable=disable_tqdm, total=max_batches if max_batches else len(dataloader)):
            if max_batches is not None and i >= max_batches:
                break

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

# === [新增] 可视化调试函数 ===
def visualize_loss_mask(batch, processor, config, step):
    """
    打印当前 Batch 中第一个样本的完整对话，并标注每个片段计算了什么 Loss。
    逻辑尽量复刻 forward 中的判断。
    """
    # 只在主进程打印
    if dist.is_initialized() and dist.get_rank() != 0:
        return

    print(f"\n{'='*20} [Debug Visualization Step {step}] {'='*20}")
    
    # 取 Batch 中的第一个样本
    # 注意：此时 batch 数据还在 CPU (因为该函数在移到 GPU 之前或之后调用皆可，这里假设传入的是 GPU tensor 则需转 CPU)
    def to_cpu(t):
        return t.cpu().tolist() if isinstance(t, torch.Tensor) else t

    input_ids = to_cpu(batch['input_ids'][0])
    labels = to_cpu(batch['labels'][0])
    
    # 获取是否含图标记 (dataset 中返回的是 tensor 或 bool)
    has_visual_t = batch['has_visual_features']
    if isinstance(has_visual_t, torch.Tensor):
        has_visual = has_visual_t[0].item()
    else:
        has_visual = has_visual_t[0]

    # 收集特征 Token ID 集合，用于判断 Latent Loss
    feature_ids = set()
    if config.extract_token_id is not None: 
        feature_ids.add(config.extract_token_id)
    for stage in config.stages: 
        feature_ids.add(stage.token_id)

    # 缓存当前文本片段
    buffer_tokens = []
    current_type = None
    
    # 辅助打印函数
    def flush_buffer(tokens, type_str):
        if not tokens: return
        text = processor.decode(tokens, skip_special_tokens=False)
        # 移除太长的换行，保持日志整洁
        text_preview = text.replace('\n', '\\n')
        if len(text_preview) > 100: text_preview = text_preview[:] + "..."
        
        # 格式化输出: [类型] 内容
        print(f"[{type_str:^18}] | {text_preview}")
        # 如果需要查看完整文本（不带截断），可以取消下面注释
        # print(f"    -> Full: {text}")

    # 遍历 Token
    for idx, (tid, lbl) in enumerate(zip(input_ids, labels)):
        # === 判定 Loss 类型 ===
        token_type = "Unknown"
        
        if lbl == -100:
            token_type = "IGNORE (Pad/User)"
        else:
            # 有效 Label (-100 以外)
            if tid in feature_ids:
                # 是特征 Token
                loss_desc = "LATENT"
                if has_visual:
                    loss_desc += "(MSE+CE)"
                else:
                    loss_desc += "(CE Only)" # 无图时只算 CE，不算 MSE
                token_type = loss_desc
            else:
                # 是普通文本
                loss_desc = "SFT"
                # 判断 VBC (简化判断: 开启 VBC 且有图 且是文本)
                # 注意：实际代码中 vbc_target_scope 可能限制为 answer，这里为了直观，统称 SFT+VBC Candidate
                if config.lambda_vbc > 0 and has_visual:
                    loss_desc += "+VBC"
                token_type = loss_desc
        
        # === 聚合相同类型并打印 ===
        if token_type != current_type:
            flush_buffer(buffer_tokens, current_type)
            current_type = token_type
            buffer_tokens = []
        
        buffer_tokens.append(tid)
    
    # 打印最后一段
    flush_buffer(buffer_tokens, current_type)
    print(f"{'='*60}\n")


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
        print(f"Require Vision Features: {config.require_vision_features}")
        
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
    
    # === 4. 验证数据集 ===
    val_loader = None
    if len(config.train_datasets) == 1:
        if args.local_rank <= 0:
            print("[Info] 检测到 Single 模式，尝试加载验证集...")
        
        splits_to_try = ['validation', 'test']
        val_dataset = None
        
        for split_name in splits_to_try:
            val_config = copy.deepcopy(config)
            target_ds_cfg = val_config.train_datasets[0]
            target_ds_cfg.split = split_name
            
            original_feat_dir = target_ds_cfg.feature_dir
            if original_feat_dir and "train" in original_feat_dir:
                new_feat_dir = original_feat_dir.replace("train", split_name)
                if os.path.exists(new_feat_dir):
                    target_ds_cfg.feature_dir = new_feat_dir
                    if args.local_rank <= 0: print(f" -> [Auto-Fix] 验证集 feature_dir: {new_feat_dir}")
                else:
                    if args.local_rank <= 0: print(f" -> [Warning] 找不到目录 {new_feat_dir}")
                    if not config.require_vision_features:
                         target_ds_cfg.feature_dir = ""
            
            try:
                temp_ds = LatentReasoningDataset(processor, val_config)
                if len(temp_ds) > 0:
                    val_dataset = temp_ds
                    if args.local_rank <= 0: print(f" -> 成功加载 '{split_name}' 集，共 {len(val_dataset)} 条样本。")
                    break
            except Exception as e:
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
    # 安全读取 debug_print_steps，默认 0 (关闭)
    debug_steps = getattr(config, 'debug_print_steps', 30)
    
    for epoch in range(config.epochs):
        if sampler: sampler.set_epoch(epoch)
        
        model_engine.train()
        pbar = tqdm(enumerate(dataloader), total=len(dataloader), disable=(args.local_rank > 0))
        
        for step, batch in pbar:
            global_step += 1
            
            # === [新增] 调用可视化 ===
            # 在数据移到 GPU 之前或之后都可以，这里放在 GPU 移动前，方便函数内处理 CPU 转换
            if debug_steps > 0 and global_step % debug_steps == 0:
                # 注意：这里直接传 batch，函数内部会处理
                visualize_loss_mask(batch, processor, config, global_step)

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