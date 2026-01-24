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
        
        self.extract_str = config.extract_token * config.extract_count
        self.reasoning_str = ""
        for stage in config.stages:
            self.reasoning_str += (stage.token * stage.count)
            
        self.max_pixels = 768*768
        
        self.mixed_data = []
        
        # 独立的随机生成器
        rng = random.Random(config.seed)
        
        print(f"--- Initializing Dataset (Seed: {config.seed}) ---")
        
        for ds_cfg in config.train_datasets:
            print(f"Loading: {ds_cfg.name} | Split: {ds_cfg.split} | Count: {ds_cfg.count}")
            try:
                hf_ds = load_dataset(ds_cfg.name, split=ds_cfg.split)
                
                total_len = len(hf_ds)
                indices = list(range(total_len))
                rng.shuffle(indices)
                
                if ds_cfg.count > 0 and ds_cfg.count < total_len:
                    indices = indices[:ds_cfg.count]
                
                print(f" -> Selected {len(indices)} samples.")

                ds_type = self._detect_type(ds_cfg.name)
                
                for idx in indices:
                    # 【关键修复】: 文件名 ID 必须与 preprocess_features.py 生成的一致
                    # 预处理脚本生成的是 "{split}_{idx}.pt"，不包含数据集前缀
                    # 路径隔离依靠 ds_cfg.feature_dir 实现
                    file_id = f"{ds_cfg.split}_{idx}" 
                    
                    self.mixed_data.append({
                        "raw_item": hf_ds[int(idx)],
                        "cfg": ds_cfg,
                        "file_id": file_id, # 用于读取特征文件
                        "ds_type": ds_type
                    })
                    
            except Exception as e:
                print(f"[Error] Failed to load {ds_cfg.name}: {e}")
        
        rng.shuffle(self.mixed_data)
        print(f"--- Total Samples Ready: {len(self.mixed_data)} ---")

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
            else:
                return Image.new('RGB', (224, 224), (0, 0, 0))
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
        last_special_pos = -1
        for i in range(len(input_ids) - 1, -1, -1):
            if input_ids[i].item() in all_special_ids:
                last_special_pos = i
                break
        
        if last_special_pos != -1 and last_special_pos + 1 < len(input_ids):
            labels[last_special_pos + 1:] = input_ids[last_special_pos + 1:]

        pixel_values = inputs.pixel_values.squeeze(0)
        image_grid_thw = inputs.image_grid_thw.squeeze(0)
        if image_grid_thw.ndim == 1: image_grid_thw = image_grid_thw.unsqueeze(0)

        # === 加载对齐特征 ===
        feature_dir = item_wrapper['cfg'].feature_dir
        alignment_dict = {}
        
        if feature_dir:
            # 使用 file_id ("{split}_{idx}") 读取文件
            feature_path = os.path.join(feature_dir, f"{item_wrapper['file_id']}.pt")
            if os.path.exists(feature_path):
                try:
                    alignment_dict = torch.load(feature_path, map_location='cpu', weights_only=True)
                except:
                    pass
        
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
            # 只有当Batch里每个样本都有这个特征时，才进行Stack
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