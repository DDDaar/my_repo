import torch
import os
from torch.utils.data import Dataset
from PIL import Image

class LatentReasoningDataset(Dataset):
    def __init__(self, data_list, processor, config, feature_dir):
        self.data = data_list
        self.processor = processor
        self.config = config
        self.feature_dir = feature_dir
        self.type1_str = "".join([f"{config.type1_base_token}{i}|>" for i in range(config.type1_count)])

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        item_id = item['id']

        image = item['image'].convert("RGB")

        choices_str = f" Choices: {', '.join(item['choices'])}." if item['choices'] else ""
        prompt_text = f"Question: {item['question']}{choices_str}"
        answer_text = item['choices'][item['answer']] if item['choices'] else str(item['answer'])

        full_text = f"{prompt_text}{self.type1_str}{self.config.latent_start_token}{answer_text}"

        # 不在这里进行 padding，交给 collate_fn 处理
        inputs = self.processor(text=[full_text], images=[image], return_tensors="pt", padding=False)

        input_ids = inputs.input_ids.squeeze(0)

        labels = input_ids.clone()
        latent_pos = (input_ids == self.config.latent_start_id).nonzero(as_tuple=True)[0]
        if len(latent_pos) > 0:
            labels[:latent_pos[-1] + 1] = -100

        # --- 修复维度处理的关键修改 ---
        # pixel_values 形状为 [Patches, Hidden]，直接 squeeze(0) 即可
        pixel_values = inputs.pixel_values.squeeze(0)

        # 【核心修复】：不要使用 squeeze/unsqueeze 的组合猜测
        # 直接使用 reshape(-1, 3) 强制将 tensor 变为 [N, 3] 的二维形状
        # 即使原本是 [3] (1维)，也会变成 [1, 3] (2维)
        # 即使原本是 [1, 1, 3]，也会变成 [1, 3]
        image_grid_thw = inputs.image_grid_thw.reshape(-1, 3).to(torch.long)

        feature_path = os.path.join(self.feature_dir, f"{item_id}.pt")
        if not os.path.exists(feature_path):
            raise FileNotFoundError(f"Feature file missing for ID: {item_id}")

        alignment_dict = torch.load(feature_path, map_location='cpu')

        return {
            "input_ids": input_ids,
            "labels": labels,
            "pixel_values": pixel_values,
            "image_grid_thw": image_grid_thw,
            "alignment_features": alignment_dict
        }


def collate_fn(batch):
    from torch.nn.utils.rnn import pad_sequence

    input_ids = pad_sequence([item['input_ids'] for item in batch], batch_first=True, padding_value=0)
    labels = pad_sequence([item['labels'] for item in batch], batch_first=True, padding_value=-100)

    # Qwen2-VL 核心：将 batch 内所有图的 patches 在 0 维拼接
    pixel_values = torch.cat([item['pixel_values'] for item in batch], dim=0)
    
    # 修复后，这里每个 item['image_grid_thw'] 都是 [1, 3]
    # cat 之后会变成 [Batch_Size, 3]，这正是模型所需要的
    image_grid_thw = torch.cat([item['image_grid_thw'] for item in batch], dim=0)

    alignment_features = {}
    for k in batch[0]['alignment_features'].keys():
        if isinstance(batch[0]['alignment_features'][k], torch.Tensor):
            alignment_features[k] = torch.stack([item['alignment_features'][k] for item in batch])
        else:
            alignment_features[k] = [item['alignment_features'][k] for item in batch]

    return {
        "input_ids": input_ids,
        "labels": labels,
        "pixel_values": pixel_values,
        "image_grid_thw": image_grid_thw,
        "alignment_features": alignment_features
    }