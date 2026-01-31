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
        
        # === 1. 构建完整的思维链模板 (带 Count 判断) ===
        
        # Extract 部分
        self.extract_str = ""
        # 只有当 count > 0 时才添加前缀和 Token
        if config.extract_count > 0:
            if config.extract_text_prefix:
                self.extract_str += config.extract_text_prefix
            self.extract_str += (config.extract_token * config.extract_count)
        
        # Reasoning Stages 部分
        self.reasoning_str = ""
        for stage in config.stages:
            # 只有当 count > 0 时才添加前缀和 Token
            if stage.count > 0:
                if stage.text_prefix:
                    self.reasoning_str += stage.text_prefix
                self.reasoning_str += (stage.token * stage.count)
            
        self.max_pixels = 768*768
        self.mixed_data = []
        
        rng = random.Random(config.seed)
        print(f"--- Initializing Dataset (Seed: {config.seed}) ---")
        
        for ds_cfg in config.train_datasets:
            print(f"Loading: {ds_cfg.name} | Split: {ds_cfg.split}")
            try:
                hf_ds = load_dataset(ds_cfg.name, split=ds_cfg.split)
                total_len = len(hf_ds)
                all_indices = list(range(total_len))
                rng.shuffle(all_indices)
                
                current_count = 0
                target_count = ds_cfg.count if ds_cfg.count > 0 else total_len
                ds_type = self._detect_type(ds_cfg.name)
                
                for idx in all_indices:
                    if current_count >= target_count: break
                    
                    file_id = f"{ds_cfg.split}_{idx}" 
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
                print(f" -> Valid samples: {current_count}")
            except Exception as e:
                print(f"[Error] Failed to load {ds_cfg.name}: {e}")
        
        rng.shuffle(self.mixed_data)
        print(f"--- Total Samples: {len(self.mixed_data)} ---")

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
        
        # 1. 生成 User Prompt
        user_prompt = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        
        # 2. 构造 Assistant 回复的各个部分
        # extract_str, reasoning_str 已经在 __init__ 里根据 count>0 处理好了
        assistant_prefix = f"{self.extract_str}{self.reasoning_str}"
        assistant_full = f"{assistant_prefix}{answer_text}"
        
        # 3. 拼接全文
        full_text = user_prompt + assistant_full
        
        # 用于定位答案起始位置的辅助文本 (User + 废话前缀)
        text_until_answer = user_prompt + assistant_prefix

        import time
        print(full_text)
        time.sleep(1999)

        # 4. Tokenize 全文
        inputs = self.processor(
            text=[full_text], 
            images=[image], 
            return_tensors="pt", 
            padding=False,
            max_pixels=self.max_pixels,
        )
        input_ids = inputs.input_ids.squeeze(0)
        attention_mask = inputs.attention_mask.squeeze(0)
        
        # === [新增] 计算 Answer Only Mask ===
        # 先 Tokenize "直到答案之前" 的部分，获取其长度
        inputs_prefix = self.processor(
            text=[text_until_answer], 
            images=[image], 
            return_tensors="pt", 
            padding=False,
            max_pixels=self.max_pixels 
        )
        prefix_len = inputs_prefix.input_ids.shape[1]
        
        answer_only_mask = torch.zeros_like(input_ids, dtype=torch.bool)
        if prefix_len < len(input_ids):
            # 从 prefix_len 开始到结束，都是真正的答案
            answer_only_mask[prefix_len:] = True

        # 5. 设置 Labels
        labels = input_ids.clone()
        
        # 获取 user_prompt 长度
        # 重新 tokenize 以确保准确匹配 processor 逻辑
        user_inputs = self.processor(text=[user_prompt], images=[image], return_tensors="pt", padding=False, max_pixels=self.max_pixels)
        user_len = user_inputs.input_ids.shape[1]
        
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
                alignment_dict = torch.load(feature_path, map_location='cpu', weights_only=True)
            except Exception as e:
                print(f"Warning: Corrupt feature file {feature_path}: {e}")
        
        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
            "pixel_values": pixel_values,
            "image_grid_thw": image_grid_thw,
            "alignment_features": alignment_dict,
            "answer_only_mask": answer_only_mask # 新增字段
        }

def collate_fn(batch):
    from torch.nn.utils.rnn import pad_sequence
    
    input_ids = pad_sequence([item['input_ids'] for item in batch], batch_first=True, padding_value=0)
    attention_mask = pad_sequence([item['attention_mask'] for item in batch], batch_first=True, padding_value=0)
    labels = pad_sequence([item['labels'] for item in batch], batch_first=True, padding_value=-100)
    
    # [新增] Padding answer_only_mask
    answer_only_mask = pad_sequence([item['answer_only_mask'] for item in batch], batch_first=True, padding_value=False)
    
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
        "alignment_features": alignment_features,
        "answer_only_mask": answer_only_mask # 传递给 Model
    }