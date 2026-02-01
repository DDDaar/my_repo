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
    
    labels_map = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J']

    if ds_type == "m3cot":
        prompt_parts = []
        if item.get('context'): prompt_parts.append(f"Context: {item['context']}")
        prompt_parts.append(f"Question: {item['question']}")
        
        # Options
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
        
        ground_truth = f"{rationale}\nAnswer: {ans_str_formatted}"

    elif ds_type == "llava":
        for turn in item['conversations']:
            if turn['from'] == 'human':
                prompt_text = turn['value'].replace("<image>", "").strip()
            elif turn['from'] == 'gpt':
                ground_truth = turn['value']
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
        
        # Answer
        ans_idx = int(item['answer'])
        if item.get('choices') and ans_idx < len(item['choices']) and ans_idx < len(labels_map):
            ground_truth = f"{labels_map[ans_idx]}. {item['choices'][ans_idx]}"
        else:
            ground_truth = str(ans_idx)

    return prompt_text, ground_truth

def get_training_sample(config):
    """从 Config 中加载第一个数据集的第一条样本"""
    if not config.train_datasets:
        raise ValueError("Config defines no training datasets!")
    
    # 取第一个数据集配置
    ds_cfg = config.train_datasets[0]
    print(f"\n[Data] Loading first sample from: {ds_cfg.name} (Split: {ds_cfg.split})")
    
    # 加载数据集 (streaming=False 确保我们可以直接取下标)
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
    parser.add_argument("--device", type=str, default="cuda:7")
    # 新增 k1, k2 参数用于指定推理范围
    parser.add_argument("--k1", type=int, default=10, help="Start index of samples")
    parser.add_argument("--k2", type=int, default=20, help="End index of samples")
    args = parser.parse_args()

    # 1. 加载配置
    print(f"Loading Config from {args.config}...")
    config = LatentConfig.load(args.config)
    
    # 2. 加载 Processor
    print(f"Loading Processor from {args.checkpoint}...")
    processor = Qwen2_5_VLProcessor.from_pretrained(args.checkpoint)
    
        
    # 3. 加载模型
    print("Loading Model...")
    model = LatentReasoningQwen(config)
    model.base_model = model.base_model.from_pretrained(args.checkpoint, torch_dtype=torch.bfloat16)

    # 加载 Projector
    proj_path = os.path.join(args.checkpoint, "projectors.bin")
    if os.path.exists(proj_path):
        model.projectors.load_state_dict(torch.load(proj_path, map_location="cpu"))
        print("Projectors loaded.")
    else:
        print("Warning: Projectors binary not found, Latent layers might be random initialized.")
    
    model.to(args.device)
    model.eval()

    # 4. 获取数据集 (优先从 eval_datasets 加载，如果没有则用 train_datasets)
    ds_cfg = config.eval_datasets[0] if hasattr(config, 'eval_datasets') and config.eval_datasets else config.train_datasets[0]
    print(f"\n[Data] Loading dataset: {ds_cfg.name} (Split: {ds_cfg.split}) for inference range [{args.k1}, {args.k2}]")
    
    try:
        dataset = load_dataset(ds_cfg.name, split=ds_cfg.split)
    except Exception as e:
        raise RuntimeError(f"Failed to load dataset {ds_cfg.name}: {e}")
    
    ds_type = detect_type(ds_cfg.name)
    results_summary = []

    # 5. 循环推理 k1 到 k2 条数据
    for idx in range(args.k1, args.k2 + 1):
        if idx >= len(dataset):
            print(f"Warning: Index {idx} out of range for dataset of size {len(dataset)}. Stopping.")
            break

        print(f"\n>>> Processing Sample {idx} <<<")
        raw_item = dataset[idx]
        image_obj = load_sample_image(raw_item, ds_cfg)
        question_text, ground_truth = format_sample_text(raw_item, ds_type)

        # 构造输入
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

        # 生成
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
        
        print(f"Input: {question_text[:]}...")
        print(f"GT:    {ground_truth}")
        print(f"Pred:  {output_text}")
        
        results_summary.append({
            "idx": idx,
            "question": question_text,
            "ground_truth": ground_truth,
            "prediction": output_text
        })

    # 6. 汇总打印
    print("\n" + "="*50)
    print(f"INFERENCE SUMMARY (Total: {len(results_summary)} samples)")
    print("="*50)
    for res in results_summary:
        # 先在外部处理好换行符替换，避免在 f-string 内部使用反斜杠
        pred_display = res['prediction'][:].replace('\n', ' ')
        gt_display = res['ground_truth'][:].replace('\n', ' ')
        
        print(f"[{res['idx']}] GT: {gt_display} | PRED: {pred_display}")
    print("="*50)

if __name__ == "__main__":
    main()