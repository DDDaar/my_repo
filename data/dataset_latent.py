import torch
import os
from torch.utils.data import Dataset

class LatentReasoningDataset(Dataset):
    def __init__(self, data_list, processor, config, feature_dir):
        self.data = data_list
        self.processor = processor
        self.config = config
        self.feature_dir = feature_dir
        
        self.extract_str = config.extract_token * config.extract_count
        self.reasoning_str = ""
        for stage in config.stages:
            self.reasoning_str += (stage.token * stage.count)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        item_id = item['id']
        image = item['image'].convert("RGB")

        choices_str = f" Choices: {', '.join(item['choices'])}." if item['choices'] else ""
        prompt_text = f"Question: {item['question']}{choices_str}"
        answer_text = item['choices'][item['answer']] if item['choices'] else str(item['answer'])

        # 构造 Messages
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image},
                    {"type": "text", "text": prompt_text},
                ],
            }
        ]
        
        base_prompt = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        full_text = f"{base_prompt}{self.extract_str}{self.reasoning_str}{answer_text}"

        # === 不限制分辨率，保持原图逻辑 ===
        inputs = self.processor(
            text=[full_text], 
            images=[image], 
            return_tensors="pt", 
            padding=False
        )

        input_ids = inputs.input_ids.squeeze(0)
        attention_mask = inputs.attention_mask.squeeze(0)
        
        # 处理 Labels
        labels = torch.full_like(input_ids, -100)
        all_special_ids = [self.config.extract_token_id] + [s.token_id for s in self.config.stages]
        
        last_pad_idx = -1
        for i in range(len(input_ids) - 1, -1, -1):
            if input_ids[i].item() in all_special_ids:
                last_pad_idx = i
                break
        
        if last_pad_idx != -1 and last_pad_idx + 1 < len(input_ids):
            labels[last_pad_idx + 1:] = input_ids[last_pad_idx + 1:]

        pixel_values = inputs.pixel_values.squeeze(0)
        image_grid_thw = inputs.image_grid_thw.squeeze(0)
        if image_grid_thw.ndim == 1: image_grid_thw = image_grid_thw.unsqueeze(0)

        feature_path = os.path.join(self.feature_dir, f"{item_id}.pt")
        if os.path.exists(feature_path):
            alignment_dict = torch.load(feature_path, map_location='cpu', weights_only=True)
        else:
            alignment_dict = {} 

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
            "pixel_values": pixel_values,
            "image_grid_thw": image_grid_thw,
            "alignment_features": alignment_dict
        }

def collate_fn(batch):
    from torch.nn.utils.rnn import pad_sequence
    
    input_ids = pad_sequence([item['input_ids'] for item in batch], batch_first=True, padding_value=0)
    attention_mask = pad_sequence([item['attention_mask'] for item in batch], batch_first=True, padding_value=0)
    labels = pad_sequence([item['labels'] for item in batch], batch_first=True, padding_value=-100)
    
    pixel_values = torch.cat([item['pixel_values'] for item in batch], dim=0)
    image_grid_thw = torch.cat([item['image_grid_thw'] for item in batch], dim=0)

    alignment_features = {}
    if batch and 'alignment_features' in batch[0] and batch[0]['alignment_features']:
        for k in batch[0]['alignment_features'].keys():
            first_val = batch[0]['alignment_features'][k]
            # 区分 Tensor 和其他类型 (如 ID 字符串)
            if isinstance(first_val, torch.Tensor):
                alignment_features[k] = torch.stack([item['alignment_features'][k] for item in batch])
            else:
                alignment_features[k] = [item['alignment_features'][k] for item in batch]

    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "labels": labels,
        "pixel_values": pixel_values,
        "image_grid_thw": image_grid_thw,
        "alignment_features": alignment_features
    }