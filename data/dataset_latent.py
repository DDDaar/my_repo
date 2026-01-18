import torch
from torch.utils.data import Dataset
from PIL import Image


import torch
import os
from torch.utils.data import Dataset
from PIL import Image

class LatentReasoningDataset(Dataset):
	def __init__(self, data_list, processor, config, feature_dir):
		self.data = data_list # 这里的 data_list 是已经注入了 id 的 HuggingFace Dataset
		self.processor = processor
		self.config = config
		self.feature_dir = feature_dir
		self.type1_str = "".join([f"{config.type1_base_token}{i}|>" for i in range(config.type1_count)])

	def __getitem__(self, idx):
		item = self.data[idx]
		item_id = item['id'] # 此时 item 已包含人工生成的 ID
		
		# 加载图片
		image = item['image'].convert("RGB")

		# 构造 ScienceQA 文本：问题 + 选项 (如有)
		# 注意：ScienceQA 的答案通常是整数索引，这里需要转为具体文本或保留逻辑
		choices_str = f" Choices: {', '.join(item['choices'])}." if item['choices'] else ""
		prompt_text = f"Question: {item['question']}{choices_str}"
		
		# 假设 answer 字段我们需要的是其索引对应的文本（或根据你的任务逻辑调整）
		# 这里简单演示直接用数值或对应的选项文本
		answer_text = item['choices'][item['answer']] if item['choices'] else str(item['answer'])

		# [cite_start]构造完整输入文本 [cite: 2]
		full_text = f"{prompt_text}{self.type1_str}{self.config.latent_start_token}{answer_text}"

		inputs = self.processor(text=[full_text], images=[image], return_tensors="pt", padding=False)
		input_ids = inputs.input_ids.squeeze(0)

		# 构造 Labels
		labels = input_ids.clone()
		latent_pos = (input_ids == self.config.latent_start_id).nonzero(as_tuple=True)[0]
		if len(latent_pos) > 0:
			labels[:latent_pos[-1] + 1] = -100

		# 【对齐加载】：使用生成的 ID 寻找特征文件
		feature_path = os.path.join(self.feature_dir, f"{item_id}.pt")
		if not os.path.exists(feature_path):
			raise FileNotFoundError(f"Feature file missing for ID: {item_id}")
			
		alignment_dict = torch.load(feature_path, map_location='cpu')

		return {
			"input_ids": input_ids,
			"labels": labels,
			"pixel_values": inputs.pixel_values,
			"image_grid_thw": inputs.image_grid_thw,
			"alignment_features": alignment_dict
		}
# collate_fn 保持不变


def collate_fn(batch):
	from torch.nn.utils.rnn import pad_sequence
	input_ids = pad_sequence([item['input_ids'] for item in batch], batch_first=True, padding_value=0)
	labels = pad_sequence([item['labels'] for item in batch], batch_first=True, padding_value=-100)

	return {
		"input_ids": input_ids,
		"labels": labels,
		"pixel_values": torch.cat([item['pixel_values'] for item in batch]),
		"image_grid_thw": torch.cat([item['image_grid_thw'] for item in batch]),
		"alignment_features": {k: torch.stack([item['alignment_features'][k] for item in batch])
		                       for k in batch[0]['alignment_features'].keys()}
	}