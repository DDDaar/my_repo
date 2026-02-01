import torch
import os
import random
from PIL import Image
from torch.utils.data import Dataset
from datasets import load_dataset
from time import sleep

class LatentReasoningDataset(Dataset):
    def __init__(self, processor, config):
        self.processor = processor
        self.config = config
        
        # === 1. 预构建思维链内部内容 (Thought Content) ===
        # 结构: prefix <|anchor|> tokens <|anchor|>
        self.thought_content = ""
        
        # 检查是否需要生成 latent 部分
        has_latent = (config.extract_count > 0) or any(s.count > 0 for s in config.stages)
        
        if has_latent:
            # Extract 阶段
            if config.extract_count > 0:
                if config.extract_text_prefix:
                    self.thought_content += config.extract_text_prefix
                if config.use_anchor:
                    self.thought_content += config.anchor_start
                self.thought_content += (config.extract_token * config.extract_count)
                if config.use_anchor:
                    self.thought_content += config.anchor_end
            
            # Reasoning Stages 阶段
            for stage in config.stages:
                if stage.count > 0:
                    if stage.text_prefix:
                        self.thought_content += stage.text_prefix
                    if config.use_anchor:
                        self.thought_content += config.anchor_start
                    self.thought_content += (stage.token * stage.count)
                    if config.use_anchor:
                        self.thought_content += config.anchor_end
            
        self.max_pixels = 768*768
        self.mixed_data = []
        
        rng = random.Random(config.seed)
        print(f"--- Initializing Dataset (Seed: {config.seed}, Require Features: {config.require_vision_features}) ---")
        
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
                    
                    # 检查特征文件是否存在
                    has_feature = False
                    if ds_cfg.feature_dir and os.path.exists(feature_path):
                        has_feature = True
                    
                    # 逻辑修改：
                    # 如果 require_vision_features 为 True，则严格过滤，不存在就 continue
                    # 如果 require_vision_features 为 False，则不存在也加入，但标记 feature_path = None
                    if config.require_vision_features:
                        if not has_feature:
                            continue
                    else:
                        if not has_feature:
                            feature_path = None # 标记为无特征

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
        
        # 标准选项标签
        labels_map = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J']

        if ds_type == "m3cot":
            prompt_parts = []
            # Context
            if item.get('context'): 
                prompt_parts.append(f"Context: {item['context']}")
            
            # Question
            prompt_parts.append(f"Question: {item['question']}")
            
            # Options (Formatted vertically with letters)
            if item.get('choices'):
                prompt_parts.append("Options:")
                for i, c in enumerate(item['choices']):
                    if i < len(labels_map):
                        prompt_parts.append(f"{labels_map[i]}. {c}")
            
            prompt_text = "\n".join(prompt_parts)
            
            # Answer
            raw_ans = item.get('answer', '')
            rationale = item.get('rationale', '')
            
            ans_str_formatted = raw_ans
            if item.get('choices'):
                try:
                    idx = item['choices'].index(raw_ans)
                    if idx < len(labels_map):
                        ans_str_formatted = f"{labels_map[idx]}. {raw_ans}"
                except ValueError:
                    pass

            answer_text = f"{rationale}\nAnswer: {ans_str_formatted}"

        elif ds_type == "llava":
            for turn in item['conversations']:
                if turn['from'] == 'human':
                    prompt_text = turn['value'].replace("<image>", "").strip()
                elif turn['from'] == 'gpt':
                    answer_text = turn['value']
                    break
        else: 
            # ScienceQA
            prompt_parts = []
            if item.get('hint'):
                prompt_parts.append(f"{item['hint']}")
            prompt_parts.append(f"Question: {item['question']}")
            if item.get('choices'):
                prompt_parts.append("Options:")
                for i, c in enumerate(item['choices']):
                    if i < len(labels_map):
                        prompt_parts.append(f"{labels_map[i]}. {c}")
                prompt_parts.append("Please select the correct answer from the options above.")
            prompt_text = "\n".join(prompt_parts)
            
            ans_idx = int(item['answer'])
            if item.get('choices') and ans_idx < len(item['choices']) and ans_idx < len(labels_map):
                answer_text = f"{labels_map[ans_idx]}. {item['choices'][ans_idx]}"
            else:
                answer_text = str(ans_idx)

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

        feature_path = item_wrapper.get('feature_path')
        has_visual_features = (feature_path is not None)

        # === 1. 构造标准 Messages ===
        full_assistant_content = ""
        
        # 核心逻辑修改：只有当存在视觉特征时，才插入 Latent Thought 过程
        # 如果没有特征，则只训练文本问答，不进行隐式推理
        if has_visual_features and self.thought_content:
            full_assistant_content += f"{self.config.think_start}{self.thought_content}{self.config.think_end}\n"
            
        full_assistant_content += f"{self.config.answer_start}{answer_text}{self.config.answer_end}"

        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image},
                    {"type": "text", "text": prompt_text},
                ],
            },
            {
                "role": "assistant",
                "content": full_assistant_content
            }
        ]
        
        # === 2. Tokenize ===
        full_text = self.processor.apply_chat_template(messages, tokenize=False)
        
        inputs = self.processor(
            text=[full_text], 
            images=[image], 
            return_tensors="pt", 
            padding=False,
            max_pixels=self.max_pixels,
        )
        input_ids = inputs.input_ids.squeeze(0)
        attention_mask = inputs.attention_mask.squeeze(0)
        
        # === 3. Labels (Mask User) ===
        user_only_msg = [messages[0]]
        user_prompt_str = self.processor.apply_chat_template(user_only_msg, tokenize=False, add_generation_prompt=True)
        user_inputs = self.processor(
            text=[user_prompt_str], images=[image], return_tensors="pt", padding=False, max_pixels=self.max_pixels
        )
        user_len = user_inputs.input_ids.shape[1]
        
        labels = input_ids.clone()
        if user_len < len(labels):
            labels[:user_len] = -100
        else:
            labels[:] = -100 

        # === 4. Answer Mask ===
        answer_only_mask = torch.zeros_like(input_ids, dtype=torch.bool)
        if self.config.answer_start_id is not None:
            start_indices = (input_ids == self.config.answer_start_id).nonzero(as_tuple=True)[0]
            if len(start_indices) > 0:
                start_idx = start_indices[-1]
                end_idx = len(input_ids)
                if self.config.answer_end_id is not None:
                    end_indices = (input_ids == self.config.answer_end_id).nonzero(as_tuple=True)[0]
                    valid_ends = end_indices[end_indices > start_idx]
                    if len(valid_ends) > 0:
                        end_idx = valid_ends[0]
                if start_idx + 1 < end_idx:
                    answer_only_mask[start_idx + 1 : end_idx] = True

        pixel_values = inputs.pixel_values.squeeze(0)
        image_grid_thw = inputs.image_grid_thw.squeeze(0)
        if image_grid_thw.ndim == 1: image_grid_thw = image_grid_thw.unsqueeze(0)

        # === 5. Load Alignment Features ===
        alignment_dict = {}
        if has_visual_features:
            try:
                alignment_dict = torch.load(feature_path, map_location='cpu', weights_only=True)
            except Exception as e:
                print(f"Warning: Corrupt feature file {feature_path}: {e}")
                # 加载失败视同无特征
                has_visual_features = False
                alignment_dict = {}
        
        # 如果没有特征（或加载失败），生成全0的 dummy features 以便 collate
        if not has_visual_features:
            for stage in self.config.stages:
                # 生成 dummy 向量: [dim]
                alignment_dict[stage.feature_key] = torch.zeros(stage.dim, dtype=torch.float32)

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
            "pixel_values": pixel_values,
            "image_grid_thw": image_grid_thw,
            "alignment_features": alignment_dict,
            "answer_only_mask": answer_only_mask,
            "has_visual_features": has_visual_features # 新增标记
        }

def collate_fn(batch):
    from torch.nn.utils.rnn import pad_sequence
    
    input_ids = pad_sequence([item['input_ids'] for item in batch], batch_first=True, padding_value=0)
    attention_mask = pad_sequence([item['attention_mask'] for item in batch], batch_first=True, padding_value=0)
    labels = pad_sequence([item['labels'] for item in batch], batch_first=True, padding_value=-100)
    answer_only_mask = pad_sequence([item['answer_only_mask'] for item in batch], batch_first=True, padding_value=False)
    
    pixel_values = torch.cat([item['pixel_values'] for item in batch], dim=0)
    image_grid_thw = torch.cat([item['image_grid_thw'] for item in batch], dim=0)
    
    # 堆叠 has_visual_features 标记
    has_visual_features = torch.tensor([item['has_visual_features'] for item in batch], dtype=torch.bool)

    # 堆叠 Alignment Features
    # 注意：__getitem__ 保证了即使无特征也有 dummy zeros，所以 keys 应该是齐全的
    alignment_features = {}
    if batch:
        # 获取所有可能的 keys (基于第一个样本，因为我们强制填充了)
        keys = batch[0]['alignment_features'].keys()
        for k in keys:
            if k in ['id', 'unique_id']: continue 
            tensors = [item['alignment_features'][k] for item in batch]
            if len(tensors) == len(batch):
                alignment_features[k] = torch.stack(tensors)

    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "labels": labels,
        "pixel_values": pixel_values,
        "image_grid_thw": image_grid_thw,
        "alignment_features": alignment_features,
        "answer_only_mask": answer_only_mask,
        "has_visual_features": has_visual_features
    }