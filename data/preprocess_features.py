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
TARGET_DATASET = "LightChen2333/M3CoT"
TARGET_SPLIT = "test" 

# === 必须配置：本地图片根目录 ===
# 请指向包含 coco/train2017 等子文件夹的父目录
# 示例：如果图片路径是 coco/train2017/001.jpg，而文件在 /data/images/coco/train2017/001.jpg
# 则此处填写 /data/images
IMAGE_ROOT = "/home/ma-user/work/lbx/dataset/LLaVA-CoT-100k" 

# === 子集验证开关 ===
USE_SUBSET = False     
SUBSET_SIZE = 50

# 自动生成路径
# 提取最后一个斜杠后的内容
SHORT_NAME = TARGET_DATASET.split('/')[-1]
print(f"SHORT_NAME = {SHORT_NAME}") 

OUTPUT_DIR = f"./data/data_preprocessed/{SHORT_NAME}/aligned_features_{SHORT_NAME}_{TARGET_SPLIT}"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

def pool_to_1d(hidden_state):
    """将 2D 特征图池化为 1D 向量"""
    if hidden_state.dim() == 4:
        return torch.mean(hidden_state, dim=[2, 3]).squeeze(0)
    elif hidden_state.dim() == 3:
        return torch.mean(hidden_state, dim=1).squeeze(0)
    return hidden_state.flatten()

def load_image_robust(sample_image):
    """
    健壮的图片加载函数：
    1. 如果是 PIL.Image 对象，直接返回
    2. 如果是 字符串 (路径)，拼接 IMAGE_ROOT 后加载
    """
    if isinstance(sample_image, str):
        # 处理相对路径
        full_path = os.path.join(IMAGE_ROOT, sample_image)
        if not os.path.exists(full_path):
            # 尝试打印错误但不崩溃，返回 None 由后续逻辑处理
            # print(f"Image not found: {full_path}")
            return None
        return Image.open(full_path).convert("RGB")
    
    elif isinstance(sample_image, Image.Image):
        return sample_image.convert("RGB")
    
    return None

def extract_all_features():
    print(f"Target Dataset: {TARGET_DATASET} | Split: {TARGET_SPLIT}")
    print(f"Sub-set Mode: {USE_SUBSET} (Size: {SUBSET_SIZE if USE_SUBSET else 'Full'})")
    print(f"Image Root: {IMAGE_ROOT}")
    print(f"Output Directory: {OUTPUT_DIR}")
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print(f"Loading dataset...")
    try:
        dataset = load_dataset(TARGET_DATASET, split=TARGET_SPLIT)
    except Exception as e:
        print(f"Error loading dataset: {e}")
        return

    # === 子集切片 ===
    if USE_SUBSET:
        print(f"Cutting dataset to first {SUBSET_SIZE} samples...")
        dataset = dataset.select(range(min(len(dataset), SUBSET_SIZE)))

    def map_id(x, idx):
        # 【修改点】强制与 train.py 逻辑保持一致：使用 {split}_{idx}
        # 不再检测原始 dataset 是否包含 'id' 字段，防止文件名不匹配
        return {'unique_id': f"{TARGET_SPLIT}_{idx}"}

    print("Mapping unique IDs...")
    dataset = dataset.map(map_id, with_indices=True)
    
    # 预先过滤：这里稍微麻烦一点，因为 filter 需要加载图片才能判断是否存在
    # 为了效率，我们先不过滤，在循环中跳过失败的
    total_items = len(dataset)
    print(f"Total items to process: {total_items}")

    # --- 1. DINOv2 ---
    print("\n[Stage 1/3] Extracting DINOv2...")
    model_dino = AutoModel.from_pretrained("facebook/dinov2-base").to(DEVICE)
    proc_dino = AutoImageProcessor.from_pretrained("facebook/dinov2-base")

    success_count = 0
    for i in tqdm(range(total_items), desc="DINOv2"):
        sample = dataset[i]
        item_id = sample['unique_id'] 
        save_path = os.path.join(OUTPUT_DIR, f"{item_id}.pt")
        
        # === 图片加载核心逻辑 ===
        img = load_image_robust(sample.get('image'))
        if img is None:
            continue

        inputs = proc_dino(images=img, return_tensors="pt").to(DEVICE)
        with torch.no_grad():
            outputs = model_dino(**inputs)
            feat = outputs.last_hidden_state[:, 0, :].squeeze(0).cpu()
        
        torch.save({"id": item_id, "dino_v2_global": feat}, save_path)
        success_count += 1

    del model_dino, proc_dino
    torch.cuda.empty_cache(); gc.collect()
    print(f"Successfully extracted {success_count}/{total_items} images.")

    # --- 2. Depth ---
    print("\n[Stage 2/3] Extracting Depth-Anything-V2...")
    model_depth = AutoModelForDepthEstimation.from_pretrained("depth-anything/Depth-Anything-V2-Base-hf").to(DEVICE)
    proc_depth = AutoImageProcessor.from_pretrained("depth-anything/Depth-Anything-V2-Base-hf")

    for i in tqdm(range(total_items), desc="Depth-V2"):
        sample = dataset[i]
        item_id = sample['unique_id']
        save_path = os.path.join(OUTPUT_DIR, f"{item_id}.pt")
        
        # 依赖第一步生成的文件，如果不存在说明第一步加载失败了
        if not os.path.exists(save_path): continue

        img = load_image_robust(sample.get('image')) # 重新加载图片
        if img is None: continue

        inputs = proc_depth(images=img, return_tensors="pt").to(DEVICE)
        with torch.no_grad():
            outputs = model_depth(**inputs, output_hidden_states=True)
            feat = pool_to_1d(outputs.hidden_states[-1]).cpu()
        
        try:
            data = torch.load(save_path)
            data.update({"depth_map_encoded": feat})
            torch.save(data, save_path)
        except Exception as e:
            print(f"Error saving depth for {item_id}: {e}")

    del model_depth, proc_depth
    torch.cuda.empty_cache(); gc.collect()

    # --- 3. SegFormer ---
    print("\n[Stage 3/3] Extracting SegFormer...")
    model_seg = AutoModelForSemanticSegmentation.from_pretrained("nvidia/segformer-b2-finetuned-ade-512-512").to(DEVICE)
    proc_seg = AutoImageProcessor.from_pretrained("nvidia/segformer-b2-finetuned-ade-512-512")

    for i in tqdm(range(total_items), desc="SegFormer"):
        sample = dataset[i]
        item_id = sample['unique_id']
        save_path = os.path.join(OUTPUT_DIR, f"{item_id}.pt")
        
        if not os.path.exists(save_path): continue

        img = load_image_robust(sample.get('image'))
        if img is None: continue

        inputs = proc_seg(images=img, return_tensors="pt").to(DEVICE)
        with torch.no_grad():
            outputs = model_seg(**inputs, output_hidden_states=True)
            feat = pool_to_1d(outputs.hidden_states[-1]).cpu()
        
        try:
            data = torch.load(save_path)
            data.update({"segmentation_map_encoded": feat})
            torch.save(data, save_path)
        except Exception as e:
            print(f"Error saving seg for {item_id}: {e}")

    print(f"\n[Success] Features saved to {OUTPUT_DIR}")

if __name__ == "__main__":
    extract_all_features()



# from datasets import load_dataset
# # Load the LLaVA-CoT-100k dataset
# dataset = load_dataset("Xkev/LLaVA-CoT-100k")
# # Access the training split
# train_split = dataset["train"]
# # Print an example
# print(train_split[38])            

