import os
import torch
import torch_npu.contrib.transfer_to_npu 
import gc
from PIL import Image
from tqdm import tqdm
from datasets import load_dataset
from transformers import (
    AutoImageProcessor,
    AutoModel,
    AutoModelForDepthEstimation,
    AutoModelForSemanticSegmentation 
)

# === 配置区域 ===
# 选项 1: ScienceQA
# TARGET_DATASET = "derek-thomas/ScienceQA"
# TARGET_SPLIT = "validation"

# 选项 2: M3CoT
TARGET_DATASET = "LightChen2333/M3CoT"
TARGET_SPLIT = "test"  # M3CoT 通常用 train 做训练

# 自动生成路径: ./data_preprocessed/M3CoT/aligned_features_M3CoT_train
SHORT_NAME = TARGET_DATASET.split('/')[-1]
OUTPUT_DIR = f"./data_preprocessed/{SHORT_NAME}/aligned_features_{SHORT_NAME}_{TARGET_SPLIT}"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

def pool_to_1d(hidden_state):
    if hidden_state.dim() == 4:
        return torch.mean(hidden_state, dim=[2, 3]).squeeze(0)
    elif hidden_state.dim() == 3:
        return torch.mean(hidden_state, dim=1).squeeze(0)
    return hidden_state.flatten()

def extract_all_features():
    print(f"Target Dataset: {TARGET_DATASET} | Split: {TARGET_SPLIT}")
    print(f"Output Directory: {OUTPUT_DIR}")
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print(f"Loading dataset...")
    # 加载原始数据
    full_dataset = load_dataset(TARGET_DATASET, split=TARGET_SPLIT)
    
    # 统一 ID 生成逻辑: {split}_{index}
    # 确保 M3CoT 和 ScienceQA 逻辑一致
    full_dataset = full_dataset.map(lambda x, idx: {'id': f"{TARGET_SPLIT}_{idx}"}, with_indices=True)
    
    # 过滤掉没有图片的样本
    dataset = full_dataset.filter(lambda x: x['image'] is not None)
    total_items = len(dataset)
    print(f"Total images to process: {total_items}")

    # 1. DINOv2
    print("\n[Stage 1/3] Extracting DINOv2...")
    model_dino = AutoModel.from_pretrained("facebook/dinov2-base").to(DEVICE)
    proc_dino = AutoImageProcessor.from_pretrained("facebook/dinov2-base")

    for i in tqdm(range(total_items), desc="DINOv2"):
        sample = dataset[i]
        item_id = sample['id'] 
        save_path = os.path.join(OUTPUT_DIR, f"{item_id}.pt")
        
        img = sample['image'].convert("RGB")
        inputs = proc_dino(images=img, return_tensors="pt").to(DEVICE)
        with torch.no_grad():
            outputs = model_dino(**inputs)
            feat = outputs.last_hidden_state[:, 0, :].squeeze(0).cpu()
        torch.save({"id": item_id, "dino_v2_global": feat}, save_path)

    del model_dino, proc_dino
    torch.cuda.empty_cache(); gc.collect()

    # 2. Depth 
    print("\n[Stage 2/3] Extracting Depth-Anything-V2...")
    model_depth = AutoModelForDepthEstimation.from_pretrained("depth-anything/Depth-Anything-V2-Base-hf").to(DEVICE)
    proc_depth = AutoImageProcessor.from_pretrained("depth-anything/Depth-Anything-V2-Base-hf")

    for i in tqdm(range(total_items), desc="Depth-V2"):
        sample = dataset[i]
        item_id = sample['id']
        save_path = os.path.join(OUTPUT_DIR, f"{item_id}.pt")
        
        img = sample['image'].convert("RGB")
        inputs = proc_depth(images=img, return_tensors="pt").to(DEVICE)
        with torch.no_grad():
            outputs = model_depth(**inputs, output_hidden_states=True)
            feat = pool_to_1d(outputs.hidden_states[-1]).cpu()
        
        data = torch.load(save_path)
        data.update({"depth_map_encoded": feat})
        torch.save(data, save_path)

    del model_depth, proc_depth
    torch.cuda.empty_cache(); gc.collect()

    # 3. SegFormer
    print("\n[Stage 3/3] Extracting SegFormer...")
    model_seg = AutoModelForSemanticSegmentation.from_pretrained("nvidia/segformer-b2-finetuned-ade-512-512").to(DEVICE)
    proc_seg = AutoImageProcessor.from_pretrained("nvidia/segformer-b2-finetuned-ade-512-512")

    for i in tqdm(range(total_items), desc="SegFormer"):
        sample = dataset[i]
        item_id = sample['id']
        save_path = os.path.join(OUTPUT_DIR, f"{item_id}.pt")
        
        img = sample['image'].convert("RGB")
        inputs = proc_seg(images=img, return_tensors="pt").to(DEVICE)
        with torch.no_grad():
            outputs = model_seg(**inputs, output_hidden_states=True)
            feat = pool_to_1d(outputs.hidden_states[-1]).cpu()
        
        data = torch.load(save_path)
        data.update({"segmentation_map_encoded": feat})
        torch.save(data, save_path)

    print(f"\n[Success] Features saved to {OUTPUT_DIR}")

if __name__ == "__main__":
    extract_all_features()