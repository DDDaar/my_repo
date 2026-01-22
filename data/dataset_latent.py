import torch
import os
from PIL import Image
from torch.utils.data import Dataset

class LatentReasoningDataset(Dataset):
    def __init__(self, data_list, processor, config, feature_dir):
        self.data = data_list
        self.processor = processor
        self.config = config
        self.feature_dir = feature_dir
        
        # === 获取 Image Folder 配置 ===
        # 兼容性处理：尝试从 config 对象或 config 字典中获取 image_folder
        self.image_folder = None
        if hasattr(config, 'image_folder'): # 如果你在 Config 类里加了字段
            self.image_folder = config.image_folder
        
        # 备用方案：如果 Config 类没更新，我们假设 config.data 原始字典还在 yaml 里
        # 但 LatentConfig 可能没把原始 data dict 存下来。
        # 这里为了稳健，建议直接在 config.yaml 填好，且我们在下面代码里硬编码读取逻辑，或者让 yaml loader 传递进来
        # 在这里，我们假设 config 对象已经通过某种方式传递了该信息，或者我们尝试读取
        # 实际上，修改 LatentConfig 类是最规范的，但如果不改 Config 类，我们可以这样：
        import yaml
        with open("config/config.yaml", 'r') as f:
            raw_cfg = yaml.safe_load(f)
            self.image_folder = raw_cfg.get('data', {}).get('image_folder', None)

        self.extract_str = config.extract_token * config.extract_count
        self.reasoning_str = ""
        for stage in config.stages:
            self.reasoning_str += (stage.token * stage.count)

        self.dataset_name = config.dataset_name
        self.is_m3cot = "M3CoT" in self.dataset_name
        self.is_llava = "LLaVA" in self.dataset_name
        self.max_pixels = 768*768

    def __len__(self):
        return len(self.data)

    def _format_scienceqa(self, item):
        choices_str = f" Choices: {', '.join(item['choices'])}." if item['choices'] else ""
        prompt_text = f"Question: {item['question']}{choices_str}"
        answer_text = item['choices'][item['answer']] if item['choices'] else str(item['answer'])
        return prompt_text, answer_text

    def _format_m3cot(self, item):
        prompt_parts = []
        if item.get('context'):
            prompt_parts.append(f"Context: {item['context']}")
        prompt_parts.append(f"Question: {item['question']}")
        
        if item.get('choices'):
            choices_formatted = []
            labels = ['A', 'B', 'C', 'D', 'E', 'F']
            for i, choice in enumerate(item['choices']):
                label = labels[i] if i < len(labels) else str(i)
                choices_formatted.append(f"{label}. {choice}")
            prompt_parts.append(f"Choices: {' '.join(choices_formatted)}")
        
        prompt_text = "\n".join(prompt_parts)
        rationale = item.get('rationale', '')
        final_answer = item.get('answer', '')
        answer_text = f"Thought: {rationale}\nAnswer: {final_answer}"
        return prompt_text, answer_text

    def _format_llava_cot(self, item):
        # 目前llava-cot-100k是单轮sft
        conversations = item['conversations']
        human_input = ""
        gpt_response = ""
        for turn in conversations:
            role = turn['from']
            content = turn['value']
            if role == 'human':
                human_input = content.replace("<image>", "").replace("\n<image>", "").strip()
            elif role == 'gpt':
                gpt_response = content
                break
        return human_input, gpt_response

    def _load_image(self, item):
        """
        统一处理图片加载：
        1. 如果是 PIL 对象 -> 直接转换 RGB
        2. 如果是 字符串 -> 拼接路径加载
        """
        img_raw = item.get('image')
        
        if isinstance(img_raw, Image.Image):
            return img_raw.convert("RGB")
        
        elif isinstance(img_raw, str):
            if self.image_folder is None:
                raise ValueError("Dataset contains image paths (strings), but 'image_folder' is not set in config.yaml")
            
            full_path = os.path.join(self.image_folder, img_raw)
            if not os.path.exists(full_path):
                # 训练时图片缺失是严重错误，建议报错或返回黑色图片
                # raise FileNotFoundError(f"Image file not found: {full_path}")
                print(f"Warning: Image missing at {full_path}, using black image.")
                return Image.new('RGB', (224, 224), (0, 0, 0))
            
            return Image.open(full_path).convert("RGB")
        
        else:
            # 如果没有图片 (纯文本数据)，这在多模态训练中可能不合法
            raise ValueError(f"Unknown image type or missing image: {type(img_raw)}")

    def __getitem__(self, idx):
        item = self.data[idx]
        
        # ID 获取逻辑
        if 'id' in item:
            item_id = str(item['id'])
        elif 'unique_id' in item:
            item_id = str(item['unique_id'])
        else:
            # 兜底：只有在非 train.py 流程（如单纯测试 dataset类）时才会用到
            item_id = f"{self.config.dataset_split}_{idx}"

        # item_id = f"{self.config.dataset_split}_{idx}"

        # === 核心修改：使用 _load_image 处理字符串路径 ===
        try:
            image = self._load_image(item)
        except Exception as e:
            print(f"Error loading image for ID {item_id}: {e}")
            # 返回一个伪造的样本避免 DataLoader 崩溃 (实际工程中常用 trick)
            image = Image.new('RGB', (224, 224), (0, 0, 0))
            # 这里的 text 处理也要小心，简单跳过

        # 格式化文本
        if self.is_m3cot:
            prompt_text, answer_text = self._format_m3cot(item)
        elif self.is_llava:
            prompt_text, answer_text = self._format_llava_cot(item)
        else:
            prompt_text, answer_text = self._format_scienceqa(item)

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

        inputs = self.processor(
            text=[full_text], 
            images=[image], 
            return_tensors="pt", 
            padding=False,
            max_pixels=self.max_pixels,
        )

        input_ids = inputs.input_ids.squeeze(0)
        attention_mask = inputs.attention_mask.squeeze(0)
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

        # 加载特征
        feature_path = os.path.join(self.feature_dir, f"{item_id}.pt")
        # print(f'准备加载的特征路径是{feature_path}')
        if os.path.exists(feature_path):
            alignment_dict = torch.load(feature_path, map_location='cpu', weights_only=True)
        else:
            alignment_dict = {} 
            print(f'准备加载的特征路径{feature_path}不存在')
            import time
            #time.sleep(5)
            
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