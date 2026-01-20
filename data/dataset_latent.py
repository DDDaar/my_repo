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

        self.is_m3cot = "M3CoT" in config.dataset_name
        self.max_pixels = 768*768

    def __len__(self):
        return len(self.data)

    def _format_scienceqa(self, item):
        """处理 ScienceQA 格式"""
        choices_str = f" Choices: {', '.join(item['choices'])}." if item['choices'] else ""
        prompt_text = f"Question: {item['question']}{choices_str}"
        # ScienceQA 的 answer 是 index，需要转换
        answer_text = item['choices'][item['answer']] if item['choices'] else str(item['answer'])
        return prompt_text, answer_text

    def _format_m3cot(self, item):
        """
        处理 M3CoT 格式
        Input: Context + Question + Choices
        Target: Rationale (CoT) + Answer
        """
        # 1. 构建 Prompt (Context + Question + Choices)
        prompt_parts = []
        if item.get('context'):
            prompt_parts.append(f"Context: {item['context']}")
        
        prompt_parts.append(f"Question: {item['question']}")
        
        if item.get('choices'):
            # M3CoT 的 choices 是一个 list，我们将其格式化为 A. xxx B. xxx
            choices_formatted = []
            labels = ['A', 'B', 'C', 'D', 'E', 'F']
            for i, choice in enumerate(item['choices']):
                label = labels[i] if i < len(labels) else str(i)
                choices_formatted.append(f"{label}. {choice}")
            prompt_parts.append(f"Choices: {' '.join(choices_formatted)}")
        
        prompt_text = "\n".join(prompt_parts)

        # 2. 构建 Answer (Rationale + Final Answer)
        # SFT 包含思考过程
        rationale = item.get('rationale', '')
        final_answer = item.get('answer', '')
        
        # 组合：思考过程 -> 结论
        answer_text = f"Thought: {rationale}\nAnswer: {final_answer}"
        
        return prompt_text, answer_text

    def __getitem__(self, idx):
        item = self.data[idx]
        item_id = item['id']
        image = item['image'].convert("RGB")

        # 根据数据集类型选择格式化策略
        if self.is_m3cot:
            prompt_text, answer_text = self._format_m3cot(item)
        else:
            prompt_text, answer_text = self._format_scienceqa(item)

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
        # 插入隐式推理 Token
        full_text = f"{base_prompt}{self.extract_str}{self.reasoning_str}{answer_text}"





        ##################

        inputs = self.processor(
            text=[full_text], 
            images=[image], 
            return_tensors="pt", 
            padding=False,
            max_pixels=640*640,
        )

        input_ids = inputs.input_ids.squeeze(0)
        attention_mask = inputs.attention_mask.squeeze(0)
        
        # 处理 Labels (忽略 Prompt 部分和特殊 Token)
        labels = torch.full_like(input_ids, -100)
        all_special_ids = [self.config.extract_token_id] + [s.token_id for s in self.config.stages]
        
        last_pad_idx = -1
        # 找到最后一个特殊 reasoning token 的位置
        for i in range(len(input_ids) - 1, -1, -1):
            if input_ids[i].item() in all_special_ids:
                last_pad_idx = i
                break
        
        # 只对推理 Token 之后的内容 (即 Answer/Rationale) 计算 Loss
        if last_pad_idx != -1 and last_pad_idx + 1 < len(input_ids):
            labels[last_pad_idx + 1:] = input_ids[last_pad_idx + 1:]

        pixel_values = inputs.pixel_values.squeeze(0)
        image_grid_thw = inputs.image_grid_thw.squeeze(0)
        if image_grid_thw.ndim == 1: image_grid_thw = image_grid_thw.unsqueeze(0)

        # 加载预提取的视觉特征
        feature_path = os.path.join(self.feature_dir, f"{item_id}.pt")
        if os.path.exists(feature_path):
            alignment_dict = torch.load(feature_path, map_location='cpu', weights_only=True)
        else:
            # 如果文件不存在，给一个空字典，避免报错 (实际训练应确保存在)
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