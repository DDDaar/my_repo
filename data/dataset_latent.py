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

    # --- 核心修改：解决 DistributedSampler 报错 ---
    def __len__(self):
        """返回数据集的总长度，供 DistributedSampler 和 DataLoader 使用"""
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        item_id = item['id'] # 此时 item 已包含人工生成的 ID
        
        # 加载图片
        image = item['image'].convert("RGB")

        # 构造 ScienceQA 文本：问题 + 选项 (如有)
        choices_str = f" Choices: {', '.join(item['choices'])}." if item['choices'] else ""
        prompt_text = f"Question: {item['question']}{choices_str}"
        
        # 答案逻辑保持不变
        answer_text = item['choices'][item['answer']] if item['choices'] else str(item['answer'])

        # 构造完整输入文本
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
            
        # map_location='cpu' 确保不会在加载阶段挤爆显存
        alignment_dict = torch.load(feature_path, map_location='cpu')

        return {
            "input_ids": input_ids,
            "labels": labels,
            "pixel_values": inputs.pixel_values.squeeze(0), # 移除多余的 batch 维度
            "image_grid_thw": inputs.image_grid_thw.squeeze(0),
            "alignment_features": alignment_dict
        }

def collate_fn(batch):
    from torch.nn.utils.rnn import pad_sequence
    
    # 1. 对文本序列进行补齐 (Padding)
    input_ids = pad_sequence([item['input_ids'] for item in batch], batch_first=True, padding_value=0)
    labels = pad_sequence([item['labels'] for item in batch], batch_first=True, padding_value=-100)

    # 2. 对图像相关的 Tensor 进行合并
    # Qwen2-VL 的 pixel_values 和 image_grid_thw 形状可能因图像大小不同而不同，
    # 使用 torch.cat 拼接在第一维
    pixel_values = torch.cat([item['pixel_values'] for item in batch], dim=0)
    image_grid_thw = torch.cat([item['image_grid_thw'] for item in batch], dim=0)

    # 3. 合并对齐特征 (Alignment Features)
    # 假设每个样本的字典键值对一致
    alignment_features = {}
    for k in batch[0]['alignment_features'].keys():
        # 如果是 Tensor，直接 stack
        if isinstance(batch[0]['alignment_features'][k], torch.Tensor):
            alignment_features[k] = torch.stack([item['alignment_features'][k] for item in batch])
        else:
            # 非 Tensor 类型直接列表组合
            alignment_features[k] = [item['alignment_features'][k] for item in batch]

    return {
        "input_ids": input_ids,
        "labels": labels,
        "pixel_values": pixel_values,
        "image_grid_thw": image_grid_thw,
        "alignment_features": alignment_features
    }