import torch
from torch.utils.data import Dataset
from PIL import Image


class LatentReasoningDataset(Dataset):
	def __init__(self, data_list, processor, config, feature_dir):
		self.data = data_list
		self.processor = processor
		self.config = config
		self.feature_dir = feature_dir
		self.type1_str = "".join([f"{config.type1_base_token}{i}|>" for i in range(config.type1_count)])

	def __getitem__(self, idx):
		item = self.data[idx]
		image = Image.open(item['image_path']).convert("RGB")

		# 构造完整文本输入 [cite: 2]
		full_text = f"{item['prompt']}{self.type1_str}{self.config.latent_start_token}{item['answer']}"

		inputs = self.processor(text=[full_text], images=[image], return_tensors="pt", padding=False)
		input_ids = inputs.input_ids.squeeze(0)

		# 构造 Labels：将 Prompt 和 Special Tokens 部分设为 -100
		labels = input_ids.clone()
		# 找到答案开始的位置：即 latent_start 之后
		latent_pos = (input_ids == self.config.latent_start_id).nonzero(as_tuple=True)[0]
		labels[:latent_pos + 1] = -100

		# 加载特征
		alignment_dict = torch.load(f"{self.feature_dir}/{item['id']}.pt", map_location='cpu')

		return {
			"input_ids": input_ids,
			"labels": labels,
			"pixel_values": inputs.pixel_values,
			"image_grid_thw": inputs.image_grid_thw,
			"alignment_features": alignment_dict
		}


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