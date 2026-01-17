import os
import torch
from PIL import Image
from tqdm import tqdm
from datasets import load_dataset
from transformers import AutoImageProcessor, AutoModel, AutoModelForDepthEstimation, AutoModelForSemanticSegmentation

# 配置
DATASET_NAME = "derek-thomas/ScienceQA"
OUTPUT_DIR = "./data/aligned_features"
DEVICE = "cuda"


def extract_all_features():
	os.makedirs(OUTPUT_DIR, exist_ok=True)
	dataset = load_dataset(DATASET_NAME, split="train").filter(lambda x: x['image'] is not None)

	# 初始化特征存储字典：{id: {feat_name: tensor}}
	all_features = {i: {} for i in range(len(dataset))}

	# --- 第一阶段：DINOv2 ---
	print("Stage 1: Extracting DINOv2...")
	model_dino = AutoModel.from_pretrained("facebook/dinov2-base").to(DEVICE)
	proc_dino = AutoImageProcessor.from_pretrained("facebook/dinov2-base")
	for i, item in tqdm(enumerate(dataset), total=len(dataset)):
		img = item['image'].convert("RGB")
		inputs = proc_dino(images=img, return_tensors="pt").to(DEVICE)
		with torch.no_grad():
			feat = model_dino(**inputs).last_hidden_state[:, 0, :].cpu()  # CLS token
		all_features[i]["dino_v2_global"] = feat.squeeze(0)
	del model_dino  # 释放内存

	# --- 第二阶段：Depth ---
	print("Stage 2: Extracting Depth...")
	model_depth = AutoModelForDepthEstimation.from_pretrained("LiheYoung/depth-anything-small-hf").to(DEVICE)
	proc_depth = AutoImageProcessor.from_pretrained("LiheYoung/depth-anything-small-hf")
	for i, item in tqdm(enumerate(dataset), total=len(dataset)):
		img = item['image'].convert("RGB")
		inputs = proc_depth(images=img, return_tensors="pt").to(DEVICE)
		with torch.no_grad():
			out = model_depth(**inputs, output_hidden_states=True)
			feat = torch.mean(out.hidden_states[-1], dim=[2, 3]).cpu()  # GAP
		all_features[i]["depth_map_encoded"] = feat.squeeze(0)
	del model_depth

	# --- 第三阶段：Segmentation ---
	print("Stage 3: Extracting Segmentation...")
	model_seg = AutoModelForSemanticSegmentation.from_pretrained("nvidia/segformer-b0-finetuned-ade-512-512").to(DEVICE)
	proc_seg = AutoImageProcessor.from_pretrained("nvidia/segformer-b0-finetuned-ade-512-512")
	for i, item in tqdm(enumerate(dataset), total=len(dataset)):
		img = item['image'].convert("RGB")
		inputs = proc_seg(images=img, return_tensors="pt").to(DEVICE)
		with torch.no_grad():
			out = model_seg(**inputs, output_hidden_states=True)
			feat = torch.mean(out.hidden_states[-1], dim=[2, 3]).cpu()  # GAP
		all_features[i]["segmentation_map_encoded"] = feat.squeeze(0)

		# 保存单个文件 (每张图一个 .pt)
		torch.save(all_features[i], os.path.join(OUTPUT_DIR, f"{i}.pt"))
	del model_seg


if __name__ == "__main__":
	extract_all_features()