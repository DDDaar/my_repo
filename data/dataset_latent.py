import torch
import os
from torch.utils.data import Dataset

class LatentReasoningDataset(Dataset):
    def __init__(self, data_list, processor, config, feature_dir):
        self.data = data_list
        self.processor = processor
        self.config = config
        self.feature_dir = feature_dir
        
        # 1. 构建提取层字符串 (无 Loss)
        self.extract_str = config.extract_token * config.extract_count
        
        # 2. 构建推理对齐层字符串 (有 MSE Loss)
        self.reasoning_str = ""
        for stage in config.stages:
            self.reasoning_str += (stage.token * stage.count)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        item_id = item['id']
        image = item['image'].convert("RGB")

        # 构造 Prompt
        choices_str = f" Choices: {', '.join(item['choices'])}." if item['choices'] else ""
        prompt_text = f"Question: {item['question']}{choices_str}"
        answer_text = item['choices'][item['answer']] if item['choices'] else str(item['answer'])

        # 消息格式
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image},
                    {"type": "text", "text": prompt_text},
                ],
            }
        ]
        
        # 生成 User Prompt (包含 <|vision_start|>...<|vision_end|> 和问题文本)
        base_prompt = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        
        # === 核心结构修改 ===
        # 结构: [User(Image+Text)] + [VisionExtract] + [Reasoning] + [Answer]
        full_text = f"{base_prompt}{self.extract_str}{self.reasoning_str}{answer_text}"

        # Tokenize
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
        
        # 逻辑：
        # 1. 找到推理结束的位置
        # 2. 只有 Answer 部分计算 Cross Entropy Loss
        # 3. Vision Extract 和 Reasoning Pads 的 Label 都是 -100 (因为它们通过 Backprop 或 MSE 学习)
        
        all_special_ids = [self.config.extract_token_id] + [s.token_id for s in self.config.stages]
        
        # 简单定位：找到最后一个特殊 Pad Token 的位置
        # 注意：我们需要 Answer 的 tokens 作为 labels
        # 倒序查找最后一个 pad token
        last_pad_idx = -1
        for i in range(len(input_ids) - 1, -1, -1):
            if input_ids[i].item() in all_special_ids:
                last_pad_idx = i
                break
        
        if last_pad_idx != -1 and last_pad_idx + 1 < len(input_ids):
            # Answer 开始于 last_pad_idx + 1
            labels[last_pad_idx + 1:] = input_ids[last_pad_idx + 1:]

        # 处理图像特征
        pixel_values = inputs.pixel_values.squeeze(0)
        image_grid_thw = inputs.image_grid_thw.squeeze(0)
        if image_grid_thw.ndim == 1: image_grid_thw = image_grid_thw.unsqueeze(0)

        # 加载对齐特征
        feature_path = os.path.join(self.feature_dir, f"{item_id}.pt")
        if os.path.exists(feature_path):
            alignment_dict = torch.load(feature_path, map_location='cpu', weights_only=True)
        else:
            # 训练时应报错，这里做 demo 兼容
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
    
    # 1. 基础数据堆叠
    input_ids = pad_sequence([item['input_ids'] for item in batch], batch_first=True, padding_value=0)
    attention_mask = pad_sequence([item['attention_mask'] for item in batch], batch_first=True, padding_value=0)
    labels = pad_sequence([item['labels'] for item in batch], batch_first=True, padding_value=-100)
    
    pixel_values = torch.cat([item['pixel_values'] for item in batch], dim=0)
    image_grid_thw = torch.cat([item['image_grid_thw'] for item in batch], dim=0)

    # 2. 对齐特征处理 (修复报错的核心部分)
    alignment_features = {}
    
    # 检查 batch 是否为空且包含 alignment_features
    if batch and 'alignment_features' in batch[0] and batch[0]['alignment_features']:
        # 遍历所有键 (如 'dino_v2_global', 'id', 'depth_map_encoded' 等)
        for k in batch[0]['alignment_features'].keys():
            first_val = batch[0]['alignment_features'][k]
            
            # 情况 A: 如果是 Tensor，则使用 torch.stack
            if isinstance(first_val, torch.Tensor):
                alignment_features[k] = torch.stack([item['alignment_features'][k] for item in batch])
            
            # 情况 B: 如果是其他类型 (如 'id' 是 str)，则保持为普通 List
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