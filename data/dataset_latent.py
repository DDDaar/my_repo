import torch
import os
import random
from PIL import Image
from torch.utils.data import Dataset
from datasets import load_dataset

class LatentReasoningDataset(Dataset):
    def __init__(self, processor, config):
        self.processor = processor
        self.config = config
        
        # === 1. 构建完整的思维链模板 ===
        # 格式: [ExtractPrefix] + [ExtractTokens] + [Stage1Prefix] + [Stage1Tokens] ...
        
        # Extract 部分
        self.extract_str = ""
        if config.extract_text_prefix:
            self.extract_str += config.extract_text_prefix
        self.extract_str += (config.extract_token * config.extract_count)
        
        # Reasoning Stages 部分
        self.reasoning_str = ""
        for stage in config.stages:
            if stage.text_prefix:
                self.reasoning_str += stage.text_prefix
            self.reasoning_str += (stage.token * stage.count)
            
        self.max_pixels = 768*768
        self.mixed_data = []
        
        # 随机数生成器
        rng = random.Random(config.seed)
        
        print(f"--- Initializing Dataset (Seed: {config.seed}) ---")
        
        # 加载数据集逻辑
        for ds_cfg in config.train_datasets:
            print(f"Loading: {ds_cfg.name} | Split: {ds_cfg.split} | Target Count: {ds_cfg.count}")
            try:
                hf_ds = load_dataset(ds_cfg.name, split=ds_cfg.split)
                total_len = len(hf_ds)
                
                all_indices = list(range(total_len))
                rng.shuffle(all_indices)
                
                print(f" -> Raw dataset size: {total_len}. Scanning for valid features...")
                
                current_count = 0
                target_count = ds_cfg.count if ds_cfg.count > 0 else total_len
                ds_type = self._detect_type(ds_cfg.name)
                
                for idx in all_indices:
                    if current_count >= target_count:
                        break
                    
                    file_id = f"{ds_cfg.split}_{idx}" 
                    
                    # 检查特征文件
                    feature_path = ""
                    if ds_cfg.feature_dir:
                        feature_path = os.path.join(ds_cfg.feature_dir, f"{file_id}.pt")
                    
                    if ds_cfg.feature_dir and not os.path.exists(feature_path):
                        continue
                        
                    self.mixed_data.append({
                        "raw_item": hf_ds[int(idx)],
                        "cfg": ds_cfg,
                        "file_id": file_id,
                        "ds_type": ds_type,
                        "feature_path": feature_path
                    })
                    current_count += 1
                
                print(f" -> Final valid samples loaded: {current_count}")
                    
            except Exception as e:
                print(f"[Error] Failed to load {ds_cfg.name}: {e}")
        
        rng.shuffle(self.mixed_data)
        print(f"--- Total Mix Samples Ready: {len(self.mixed_data)} ---")

    def _detect_type(self, name):
        if "M3CoT" in name: return "m3cot"
        if "LLaVA" in name: return "llava"
        return "scienceqa"

    def __len__(self):
        return len(self.mixed_data)

    def _format_text(self, item_wrapper):
        item = item_wrapper['raw_item']
        ds_type = item_wrapper['ds_type']
        
        prompt_text = ""
        answer_text = ""

        if ds_type == "m3cot":
            prompt_parts = []
            if item.get('context'): prompt_parts.append(f"Context: {item['context']}")
            prompt_parts.append(f"Question: {item['question']}")
            if item.get('choices'):
                labels = ['A', 'B', 'C', 'D', 'E', 'F']
                choices_fmt = [f"{labels[i]}. {c}" for i, c in enumerate(item['choices'])]
                prompt_parts.append(f"Choices: {' '.join(choices_fmt)}")
            prompt_text = "\n".join(prompt_parts)
            answer_text = f"Thought: {item.get('rationale','')}\nAnswer: {item.get('answer','')}"

        elif ds_type == "llava":
            for turn in item['conversations']:
                if turn['from'] == 'human':
                    prompt_text = turn['value'].replace("<image>", "").strip()
                elif turn['from'] == 'gpt':
                    answer_text = turn['value']
                    break
        else: 
            # ScienceQA
            choices = f" Choices: {', '.join(item['choices'])}." if item.get('choices') else ""
            prompt_text = f"Question: {item['question']}{choices}"
            ans_idx = item['answer']
            answer_text = item['choices'][ans_idx] if item.get('choices') else str(ans_idx)

        return prompt_text, answer_text

    def _load_image(self, item_wrapper):
        raw_item = item_wrapper['raw_item']
        ds_cfg = item_wrapper['cfg']
        img_raw = raw_item.get('image')
        
        if isinstance(img_raw, Image.Image):
            return img_raw.convert("RGB")
        elif isinstance(img_raw, str):
            if not ds_cfg.image_folder:
                return Image.new('RGB', (224, 224), (0, 0, 0))
            full_path = os.path.join(ds_cfg.image_folder, img_raw)
            if os.path.exists(full_path):
                return Image.open(full_path).convert("RGB")
        
        return Image.new('RGB', (224, 224), (0, 0, 0))

    def __getitem__(self, idx):
        item_wrapper = self.mixed_data[idx]
        
        image = self._load_image(item_wrapper)
        prompt_text, answer_text = self._format_text(item_wrapper)

        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image},
                    {"type": "text", "text": prompt_text},
                ],
            }
        ]
        
        # 1. 生成 User 部分的 Prompt (用于计算 Mask 长度)
        user_prompt = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        
        # 2. 生成 Assistant 的完整回答
        # 包含: 提取前缀 + 提取Token + 推理前缀 + 推理Token + 最终答案
        assistant_response = f"{self.extract_str}{self.reasoning_str}{answer_text}"
        
        # 3. 拼接全文
        full_text = user_prompt + assistant_response

        # 4. Tokenize
        inputs = self.processor(
            text=[full_text], 
            images=[image], 
            return_tensors="pt", 
            padding=False,
            max_pixels=self.max_pixels,
        )

        input_ids = inputs.input_ids.squeeze(0)
        attention_mask = inputs.attention_mask.squeeze(0)
        
        # 5. 设置 Labels
        labels = input_ids.clone()
        
        # 计算 user_prompt 的 token 长度
        # 重新 tokenize 以确保准确
        user_inputs = self.processor(text=[user_prompt], images=[image], return_tensors="pt", padding=False)
        user_len = user_inputs.input_ids.shape[1]
        
        # Mask 掉 User 部分，保留 Assistant 的思考过程和答案
        if user_len < len(labels):
            labels[:user_len] = -100
        else:
            labels[:] = -100 

        pixel_values = inputs.pixel_values.squeeze(0)
        image_grid_thw = inputs.image_grid_thw.squeeze(0)
        if image_grid_thw.ndim == 1: image_grid_thw = image_grid_thw.unsqueeze(0)

        # 6. 加载特征
        alignment_dict = {}
        feature_path = item_wrapper.get('feature_path')
        
        if feature_path and os.path.exists(feature_path):
            try:
                # cpu load 避免多进程 CUDA 初始化问题
                alignment_dict = torch.load(feature_path, map_location='cpu', weights_only=True)
            except Exception as e:
                print(f"Warning: Corrupt feature file {feature_path}: {e}")
        
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
    if batch and batch[0]['alignment_features']:
        keys = batch[0]['alignment_features'].keys()
        for k in keys:
            if k in ['id', 'unique_id']: continue 
            tensors = [item['alignment_features'][k] for item in batch if k in item['alignment_features']]
            if len(tensors) == len(batch):
                alignment_features[k] = torch.stack(tensors)

    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "labels": labels,
        "pixel_values": pixel_values,
        "image_grid_thw": image_grid_thw,
        "alignment_features": alignment_features
    }