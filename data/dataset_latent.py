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

    # 1. 准备图像
    image = item['image'].convert("RGB")

    # 2. 构造对话消息格式 (Qwen2.5-VL 推荐方式)
    choices_str = f" Choices: {', '.join(item['choices'])}." if item['choices'] else ""
    prompt_text = f"Question: {item['question']}{choices_str}"
    
    # 获取答案文本
    answer_text = item['choices'][item['answer']] if item['choices'] else str(item['answer'])

    # 使用 messages 格式，以便调用 chat_template 自动插入 <|vision_start|> 等标记
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": prompt_text},
            ],
        }
    ]

    # 3. 生成基础 Prompt 并拼接 Latent Reasoning 标记
    # tokenize=False 先获取带特殊标记的完整文本
    base_prompt = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    
    # 拼接推理 Token 和最终答案
    # 格式为: <|im_start|>user...<|im_end|><|im_start|>assistant\n<|type1_0|>...<|latent_start|>Answer
    full_text = f"{base_prompt}{self.type1_str}{self.config.latent_start_token}{answer_text}"

    # 4. 调用 Processor 生成模型输入
    # 此时 processor 会识别到文本中的 <|vision_start|> 并与 images 匹配
    inputs = self.processor(
        text=[full_text], 
        images=[image], 
        return_tensors="pt", 
        padding=False
    )

    input_ids = inputs.input_ids.squeeze(0)

    # 5. 构造 Labels 并屏蔽不需要计算 Loss 的部分
    labels = input_ids.clone()
    
    # 找到 <|latent_start|> 的位置，该 token 之前的所有内容（含 image tokens）均设为 -100
    latent_pos = (input_ids == self.config.latent_start_id).nonzero(as_tuple=True)[0]
    if len(latent_pos) > 0:
        # 屏蔽从开头到 <|latent_start|> 的所有 token
        labels[:latent_pos[-1] + 1] = -100
    else:
        # 回退机制：如果没找到标记，则屏蔽全部（防止报错）
        labels[:] = -100

    # 6. 处理维度
    # pixel_values 形状为 [Patches, Hidden]
    pixel_values = inputs.pixel_values.squeeze(0)
    
    # image_grid_thw 必须保持 [N, 3] 形状
    image_grid_thw = inputs.image_grid_thw.squeeze(0)
    if image_grid_thw.ndim == 1:
        image_grid_thw = image_grid_thw.unsqueeze(0)

    # 7. 加载对齐特征
    feature_path = os.path.join(self.feature_dir, f"{item_id}.pt")
    if not os.path.exists(feature_path):
        raise FileNotFoundError(f"Feature file missing for ID: {item_id}")

    alignment_dict = torch.load(feature_path, map_location='cpu', weights_only=True)

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