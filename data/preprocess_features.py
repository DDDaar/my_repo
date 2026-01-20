# import os
# import torch
# import torch_npu.contrib.transfer_to_npu  # 保持 NPU 配置
# import gc
# from PIL import Image
# from tqdm import tqdm
# from datasets import load_dataset
# from transformers import AutoImageProcessor, AutoModel, AutoModelForDepthEstimation, AutoModelForSemanticSegmentation

# # 配置
# DATASET_NAME = "derek-thomas/ScienceQA"
# OUTPUT_DIR = "./data_preprosessed/aligned_features"
# DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# def pool_feature(hidden_state):
#     """
#     自适应池化函数：
#     - 如果是 (B, C, H, W)，执行空间平均池化。
#     - 如果是 (B, L, D)，执行序列平均池化。
#     """
#     if hidden_state.dim() == 4:
#         # [B, C, H, W] -> [C]
#         return torch.mean(hidden_state, dim=[2, 3]).squeeze(0)
#     elif hidden_state.dim() == 3:
#         # [B, L, D] -> [D]
#         return torch.mean(hidden_state, dim=1).squeeze(0)
#     else:
#         return hidden_state.squeeze()

# def print_dims(stage_name, img, inputs, last_layer, final_feat):
#     """打印维度监控信息"""
#     print(f"\n{'-'*20} {stage_name} Dimension Monitor {'-'*20}")
#     print(f"1. Original Image Size : {img.size}")
#     print(f"2. Input Tensor Shape  : {inputs['pixel_values'].shape}")
#     print(f"3. Last Hidden Layer   : {last_layer.shape}")
#     print(f"4. Pooled Feature      : {final_feat.shape}")
#     print(f"{'-'*60}\n")

# def extract_all_features():
#     os.makedirs(OUTPUT_DIR, exist_ok=True)

#     # 1. 加载数据集
#     print(f"Loading dataset: {DATASET_NAME}...")
#     full_dataset = load_dataset(DATASET_NAME, split="train")
#     dataset = full_dataset.filter(lambda x: x['image'] is not None)
#     total_items = len(dataset)

#     # ==================== 第一阶段：DINOv2 (通用语义特征) ====================
#     print("\nStage 1: Extracting DINOv2 features...")
#     model_dino = AutoModel.from_pretrained("facebook/dinov2-base").to(DEVICE)
#     proc_dino = AutoImageProcessor.from_pretrained("facebook/dinov2-base")

#     for i in tqdm(range(total_items), desc="DINOv2"):
#         sample = dataset[i]
#         item_id = sample.get('id', f"idx_{i}")
#         save_path = os.path.join(OUTPUT_DIR, f"{item_id}.pt")

#         img = sample['image'].convert("RGB")
#         inputs = proc_dino(images=img, return_tensors="pt").to(DEVICE)

#         with torch.no_grad():
#             outputs = model_dino(**inputs)
#             # DINOv2 使用 [CLS] token (index 0)
#             last_layer = outputs.last_hidden_state # [B, L, D]
#             feat = last_layer[:, 0, :].squeeze(0).cpu()

#         if i == 0:
#             print_dims("DINOv2 (CLS Token)", img, inputs, last_layer, feat)

#         torch.save({"id": item_id, "dino_v2_global": feat}, save_path)

#     del model_dino, proc_dino
#     torch.cuda.empty_cache()
#     gc.collect()

#     # ==================== 第二阶段：Depth (深度结构特征) ====================
#     print("\nStage 2: Extracting Depth features...")
#     model_depth = AutoModelForDepthEstimation.from_pretrained("LiheYoung/depth-anything-small-hf").to(DEVICE)
#     proc_depth = AutoImageProcessor.from_pretrained("LiheYoung/depth-anything-small-hf")

#     for i in tqdm(range(total_items), desc="Depth"):
#         sample = dataset[i]
#         item_id = sample.get('id', f"idx_{i}")
#         save_path = os.path.join(OUTPUT_DIR, f"{item_id}.pt")

#         img = sample['image'].convert("RGB")
#         inputs = proc_depth(images=img, return_tensors="pt").to(DEVICE)

#         with torch.no_grad():
#             outputs = model_depth(**inputs, output_hidden_states=True)
#             last_layer = outputs.hidden_states[-1]
#             feat = pool_feature(last_layer).cpu()

#         if i == 0:
#             print_dims("Depth-Anything (Global Pool)", img, inputs, last_layer, feat)

#         # 稳健保存逻辑
#         data = torch.load(save_path) if os.path.exists(save_path) else {"id": item_id}
#         data.update({"depth_map_encoded": feat})
#         torch.save(data, save_path)

#     del model_depth, proc_depth
#     torch.cuda.empty_cache()
#     gc.collect()

#     # ==================== 第三阶段：Segmentation (类别分布特征) ====================
#     print("\nStage 3: Extracting Segmentation features...")
#     model_seg = AutoModelForSemanticSegmentation.from_pretrained("nvidia/segformer-b0-finetuned-ade-512-512").to(DEVICE)
#     proc_seg = AutoImageProcessor.from_pretrained("nvidia/segformer-b0-finetuned-ade-512-512")

#     for i in tqdm(range(total_items), desc="Segmentation"):
#         sample = dataset[i]
#         item_id = sample.get('id', f"idx_{i}")
#         save_path = os.path.join(OUTPUT_DIR, f"{item_id}.pt")

#         img = sample['image'].convert("RGB")
#         inputs = proc_seg(images=img, return_tensors="pt").to(DEVICE)

#         with torch.no_grad():
#             outputs = model_seg(**inputs, output_hidden_states=True)
#             last_layer = outputs.hidden_states[-1]
#             feat = pool_feature(last_layer).cpu()

#         if i == 0:
#             print_dims("SegFormer (Global Pool)", img, inputs, last_layer, feat)

#         data = torch.load(save_path) if os.path.exists(save_path) else {"id": item_id}
#         data.update({"segmentation_map_encoded": feat})
#         torch.save(data, save_path)

#     print(f"\n[Success] All features extracted and saved to {OUTPUT_DIR}")

# if __name__ == "__main__":
#     extract_all_features()


# import os
# import torch
# import torch_npu.contrib.transfer_to_npu  # 保持 NPU 配置
# import gc
# from PIL import Image
# from tqdm import tqdm
# from datasets import load_dataset
# # 引入相应的 Auto 类
# from transformers import (
#     AutoImageProcessor,
#     AutoModel,
#     AutoModelForDepthEstimation,
#     AutoModelForUniversalSegmentation # Mask2Former 使用通用分割类
# )

# # 配置
# DATASET_NAME = "derek-thomas/ScienceQA"
# OUTPUT_DIR = "./data_preprosessed/aligned_features"
# DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# def pool_feature(hidden_state):
#     """自适应池化：处理 (B, C, H, W) 或 (B, L, D)"""
#     if hidden_state.dim() == 4:
#         return torch.mean(hidden_state, dim=[2, 3]).squeeze(0)
#     elif hidden_state.dim() == 3:
#         return torch.mean(hidden_state, dim=1).squeeze(0)
#     return hidden_state.squeeze()

# def print_dims(stage_name, img, inputs, last_layer, final_feat):
#     print(f"\n{'-'*20} {stage_name} Dimension Monitor {'-'*20}")
#     print(f"1. Original Image Size : {img.size}")
#     print(f"2. Input Tensor Shape  : {inputs['pixel_values'].shape}")
#     print(f"3. Last Hidden Layer   : {last_layer.shape}")
#     print(f"4. Pooled Feature      : {final_feat.shape}")
#     print(f"{'-'*60}\n")

# def extract_all_features():
#     os.makedirs(OUTPUT_DIR, exist_ok=True)

#     print(f"Loading dataset: {DATASET_NAME}...")
#     full_dataset = load_dataset(DATASET_NAME, split="train")
#     dataset = full_dataset.filter(lambda x: x['image'] is not None)
#     total_items = len(dataset)

#     # ==================== 第一阶段：DINOv2 (保持不变，已是顶级) ====================
#     print("\nStage 1: Extracting DINOv2 (Semantic)...")
#     model_dino = AutoModel.from_pretrained("facebook/dinov2-base").to(DEVICE)
#     proc_dino = AutoImageProcessor.from_pretrained("facebook/dinov2-base")

#     for i in tqdm(range(total_items), desc="DINOv2"):
#         sample = dataset[i]
#         item_id = sample.get('id', f"idx_{i}")
#         save_path = os.path.join(OUTPUT_DIR, f"{item_id}.pt")
#         img = sample['image'].convert("RGB")
#         inputs = proc_dino(images=img, return_tensors="pt").to(DEVICE)

#         with torch.no_grad():
#             outputs = model_dino(**inputs)
#             last_layer = outputs.last_hidden_state
#             feat = last_layer[:, 0, :].squeeze(0).cpu() # CLS Token

#         if i == 0: print_dims("DINOv2", img, inputs, last_layer, feat)
#         torch.save({"id": item_id, "dino_v2_global": feat}, save_path)

#     del model_dino, proc_dino
#     torch.cuda.empty_cache(); gc.collect()

#     # ==================== 第二阶段：Depth (升级到 V2) ====================
#     print("\nStage 2: Extracting Depth-Anything-V2 (Geometry)...")
#     # V2-Small 模型在深度估计的连续性和边缘处理上更优
#     model_path = "depth-anything/Depth-Anything-V2-Small-hf"
#     model_depth = AutoModelForDepthEstimation.from_pretrained(model_path).to(DEVICE)
#     proc_depth = AutoImageProcessor.from_pretrained(model_path)

#     for i in tqdm(range(total_items), desc="Depth-V2"):
#         sample = dataset[i]
#         item_id = sample.get('id', f"idx_{i}")
#         save_path = os.path.join(OUTPUT_DIR, f"{item_id}.pt")
#         img = sample['image'].convert("RGB")
#         inputs = proc_depth(images=img, return_tensors="pt").to(DEVICE)

#         with torch.no_grad():
#             outputs = model_depth(**inputs, output_hidden_states=True)
#             # 取得最后一层隐藏层特征
#             last_layer = outputs.hidden_states[-1]
#             feat = pool_feature(last_layer).cpu()

#         if i == 0: print_dims("Depth-V2", img, inputs, last_layer, feat)
#         data = torch.load(save_path) if os.path.exists(save_path) else {"id": item_id}
#         data.update({"depth_map_encoded": feat})
#         torch.save(data, save_path)

#     del model_depth, proc_depth
#     torch.cuda.empty_cache(); gc.collect()

#     # ==================== 第三阶段：Segmentation (升级到 Mask2Former) ====================
#     print("\nStage 3: Extracting Mask2Former (Instance/Semantic)...")
#     # Mask2Former 是目前的通用分割王者，Swin-Small 骨干网络平衡了性能与速度
#     model_path = "facebook/mask2former-swin-small-ade-semantic"
#     model_seg = AutoModelForUniversalSegmentation.from_pretrained(model_path).to(DEVICE)
#     proc_seg = AutoImageProcessor.from_pretrained(model_path)

#     for i in tqdm(range(total_items), desc="Mask2Former"):
#         sample = dataset[i]
#         item_id = sample.get('id', f"idx_{i}")
#         save_path = os.path.join(OUTPUT_DIR, f"{item_id}.pt")
#         img = sample['image'].convert("RGB")
#         inputs = proc_seg(images=img, return_tensors="pt").to(DEVICE)

#         with torch.no_grad():
#             # Mask2Former 输出包含类查询（Class Queries）特征，我们取其 Backbone 的最后输出
#             outputs = model_seg(**inputs, output_hidden_states=True)
#             # hidden_states[-1] 通常是 Transformer Decoder 的输出，代表了场景中的对象查询特征
#             last_layer = outputs.hidden_states[-1]
#             feat = pool_feature(last_layer).cpu()

#         if i == 0: print_dims("Mask2Former", img, inputs, last_layer, feat)
#         data = torch.load(save_path) if os.path.exists(save_path) else {"id": item_id}
#         data.update({"segmentation_map_encoded": feat})
#         torch.save(data, save_path)

#     print(f"\n[ALT Success] All SOTA features saved to {OUTPUT_DIR}")

# if __name__ == "__main__":
#     extract_all_features()

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

DATASET_NAME = "derek-thomas/ScienceQA"
save_data_name=DATASET_NAME.split('/')[-1]
data_split = 'test'
OUTPUT_DIR = f"./data_preprocessed/{save_data_name}/aligned_features_ScienceQA_{data_split}"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

def pool_to_1d(hidden_state):
	if hidden_state.dim() == 4:
		return torch.mean(hidden_state, dim=[2, 3]).squeeze(0)
	elif hidden_state.dim() == 3:
		return torch.mean(hidden_state, dim=1).squeeze(0)
	return hidden_state.flatten()

def extract_all_features():
	os.makedirs(OUTPUT_DIR, exist_ok=True)

	print(f"Loading dataset: {DATASET_NAME}...")
	# 加载原始数据
	full_dataset = load_dataset(DATASET_NAME, split=data_split)
	
	# 【关键修改】：在过滤前为每一条数据生成唯一 ID，格式为 "train_0", "train_1"...
	# 这样即便后续过滤掉没有图片的样本，保留下来的样本 ID 依然是固定的
	full_dataset = full_dataset.map(lambda x, idx: {'id': f"{data_split}_{idx}"}, with_indices=True)
	
	# 过滤掉没有图片的样本
	dataset = full_dataset.filter(lambda x: x['image'] is not None)
	total_items = len(dataset)

	# 1. DINOv2
	model_dino = AutoModel.from_pretrained("facebook/dinov2-base").to(DEVICE)
	proc_dino = AutoImageProcessor.from_pretrained("facebook/dinov2-base")

	for i in tqdm(range(total_items), desc="DINOv2"):
		sample = dataset[i]
		item_id = sample['id'] # 获取我们生成的 ID
		save_path = os.path.join(OUTPUT_DIR, f"{item_id}.pt")
		
		img = sample['image'].convert("RGB")
		inputs = proc_dino(images=img, return_tensors="pt").to(DEVICE)
		with torch.no_grad():
			outputs = model_dino(**inputs)
			feat = outputs.last_hidden_state[:, 0, :].squeeze(0).cpu()
		torch.save({"id": item_id, "dino_v2_global": feat}, save_path)

	del model_dino, proc_dino
	torch.cuda.empty_cache(); gc.collect()

	# 2. Depth (同理使用 item_id)
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