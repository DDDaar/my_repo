import os
import torch
from torch_npu.contrib import transfer_to_npu # 若非 Ascend NPU 可注释
import argparse
from transformers import Qwen2_5_VLProcessor
from model.modeling_latent_qwen import LatentReasoningQwen
from config.configuration_latent import LatentConfig
from PIL import Image
from datasets import load_dataset

# === 复用 dataset_latent.py 的核心逻辑 ===

def detect_type(name):
    if "M3CoT" in name: return "m3cot"
    if "LLaVA" in name: return "llava"
    return "scienceqa"

def load_sample_image(item, ds_cfg):
    img_raw = item.get('image')
    
    if isinstance(img_raw, Image.Image):
        return img_raw.convert("RGB")
    elif isinstance(img_raw, str):
        # 如果是路径字符串，需要拼接 config 中的 image_folder
        if not ds_cfg.image_folder:
            # 如果没有配置图片路径，返回黑图
            return Image.new('RGB', (224, 224), (0, 0, 0))
        full_path = os.path.join(ds_cfg.image_folder, img_raw)
        if os.path.exists(full_path):
            return Image.open(full_path).convert("RGB")
    
    # 默认黑图
    return Image.new('RGB', (224, 224), (0, 0, 0))

def format_sample_text(item, ds_type):
    prompt_text = ""
    ground_truth = ""

    if ds_type == "m3cot":
        prompt_parts = []
        if item.get('context'): prompt_parts.append(f"Context: {item['context']}")
        prompt_parts.append(f"Question: {item['question']}")
        if item.get('choices'):
            labels = ['A', 'B', 'C', 'D', 'E', 'F']
            choices_fmt = [f"{labels[i]}. {c}" for i, c in enumerate(item['choices'])]
            prompt_parts.append(f"Choices: {' '.join(choices_fmt)}")
        prompt_text = "\n".join(prompt_parts)
        ground_truth = f"{item.get('rationale','')}\nAnswer: {item.get('answer','')}"

    elif ds_type == "llava":
        for turn in item['conversations']:
            if turn['from'] == 'human':
                prompt_text = turn['value'].replace("<image>", "").strip()
            elif turn['from'] == 'gpt':
                ground_truth = turn['value']
                break
    else: 
        # ScienceQA
        choices = f" Choices: {', '.join(item['choices'])}." if item.get('choices') else ""
        prompt_text = f"Question: {item['question']}{choices}"
        ans_idx = item['answer']
        ground_truth = item['choices'][ans_idx] if item.get('choices') else str(ans_idx)

    return prompt_text, ground_truth

def get_training_sample(config):
    """从 Config 中加载第一个数据集的第一条样本"""
    if not config.train_datasets:
        raise ValueError("Config defines no training datasets!")
    
    # 取第一个数据集配置
    ds_cfg = config.train_datasets[0]
    print(f"\n[Data] Loading first sample from: {ds_cfg.name} (Split: {ds_cfg.split})")
    
    # 加载数据集 (streaming=False 确保我们可以直接取下标)
    # 注意：如果网络不好，这里可能会卡住下载，建议确保本地有缓存
    try:
        dataset = load_dataset(ds_cfg.name, split=ds_cfg.split)
    except Exception as e:
        raise RuntimeError(f"Failed to load dataset {ds_cfg.name}: {e}")
    
    # 取第 0 条数据
    raw_item = dataset[0]
    ds_type = detect_type(ds_cfg.name)
    
    # 格式化
    image = load_sample_image(raw_item, ds_cfg)
    prompt_text, ground_truth = format_sample_text(raw_item, ds_type)
    
    return image, prompt_text, ground_truth

# ==========================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, required=True, help="HF format checkpoint path")
    parser.add_argument("--config", type=str, default="./config/config.yaml")
    parser.add_argument("--device", type=str, default="cuda")
    args = parser.parse_args()

    # 1. 加载配置
    print(f"Loading Config from {args.config}...")
    config = LatentConfig.load(args.config)
    
    # 2. 加载 Processor
    print(f"Loading Processor from {args.checkpoint}...")
    processor = Qwen2_5_VLProcessor.from_pretrained(args.checkpoint)
    
    # eval.py 中加载 processor 后必须有这一步
    new_tokens = [
        config.extract_token,
        config.think_start, config.think_end,
        config.answer_start, config.answer_end,
        config.anchor_start, config.anchor_end
    ] + [stage.token for stage in config.stages]

    # 这一步非常关键！
    processor.tokenizer.add_special_tokens({"additional_special_tokens": list(set(new_tokens))})

        
    # 3. 加载模型
    print("Loading Model...")
    model = LatentReasoningQwen(config)
    model.base_model = model.base_model.from_pretrained(args.checkpoint, torch_dtype=torch.bfloat16)
    model.base_model.resize_token_embeddings(len(processor.tokenizer))
    
    # 加载 Projector
    proj_path = os.path.join(args.checkpoint, "projectors.bin")
    if os.path.exists(proj_path):
        model.projectors.load_state_dict(torch.load(proj_path, map_location="cpu"))
        print("Projectors loaded.")
    else:
        print("Warning: Projectors binary not found, Latent layers might be random initialized.")
    
    model.to(args.device)
    model.eval()

    # 4. 获取训练集第一条数据
    image_obj, question_text, ground_truth = get_training_sample(config)
    
    print("-" * 30)
    print(f"Input Question: {question_text}")
    print(f"Ground Truth:   {ground_truth}")
    print("-" * 30)

    # 5. 构造输入
    messages = [
        {"role": "user", "content": [
            {"type": "image", "image": image_obj},
            {"type": "text", "text": question_text}
        ]}
    ]
    
    text_prompt = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    
    inputs = processor(
        text=[text_prompt],
        images=[image_obj],
        return_tensors="pt"
    )
    inputs = {k: v.to(args.device) for k, v in inputs.items()}
    if "image_grid_thw" in inputs:
        inputs["image_grid_thw"] = inputs["image_grid_thw"].to(args.device)

    # 6. 生成
    print("\n--- Start Autoregressive Generation ---")
    
    # 增加停止词处理，防止生成过长
    stop_words = ["<|im_end|>", "<|endoftext|>"]
    
    with torch.no_grad():
        generated_ids = model.base_model.generate(
            **inputs,
            max_new_tokens=512, 
            use_cache=True, 
            do_sample=False,
            temperature=0.5,
            top_p=0.9
        )
    
    input_len = inputs['input_ids'].shape[1]
    output_ids = generated_ids[0][input_len:]
    output_text = processor.decode(output_ids, skip_special_tokens=False)
    
    print("\n--- Generated Output ---")
    print(output_text)

if __name__ == "__main__":
    main()