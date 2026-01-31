import os
import torch
from torch_npu.contrib import transfer_to_npu # 若非 Ascend NPU 可注释
import argparse
from transformers import Qwen2_5_VLProcessor
from model.modeling_latent_qwen import LatentReasoningQwen
from config.configuration_latent import LatentConfig
from data.dataset_latent import LatentReasoningDataset
from PIL import Image


#/home/ma-user/work/lbx/models/Qwen2.5-VL-3B-Instruct

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, default="/home/ma-user/work/lbx/my_repo/checkpoints/derek-thomas_ScienceQA/epoch_0/hf_format", help="HF format checkpoint path")
    parser.add_argument("--config", type=str, default="./config/config.yaml")
    parser.add_argument("--device", type=str, default="cuda")
    args = parser.parse_args()

    # 1. 加载配置
    print(f"Loading Config from {args.config}...")
    config = LatentConfig.load(args.config)
    
    # 2. 加载 Processor 和 Model
    print(f"Loading Processor & Model from {args.checkpoint}...")
    processor = Qwen2_5_VLProcessor.from_pretrained(args.checkpoint)
    
    # 同步 Config ID
    config.extract_token_id = processor.tokenizer.convert_tokens_to_ids(config.extract_token)
    for stage in config.stages:
        stage.token_id = processor.tokenizer.convert_tokens_to_ids(stage.token)
        
    model = LatentReasoningQwen(config)
    # 加载 base_model 权重
    model.base_model = model.base_model.from_pretrained(args.checkpoint, torch_dtype=torch.bfloat16)
    
    # 加载 Projector 权重
    proj_path = os.path.join(args.checkpoint, "projectors.bin")
    if os.path.exists(proj_path):
        model.projectors.load_state_dict(torch.load(proj_path, map_location="cpu"))
        print("Projectors loaded.")
    
    model.to(args.device)
    model.eval()

    # 3. 初始化数据集并获取第一条数据
    print(f"Initializing dataset to fetch the first sample...")
    dataset = LatentReasoningDataset(processor, config)
    if len(dataset) == 0:
        print("❌ Dataset is empty. Check your config.yaml and data paths.")
        return
    
    # 获取第一条数据（包含 raw_item）
    first_item_wrapper = dataset.mixed_data[0]
    image = dataset._load_image(first_item_wrapper)
    prompt_text, ground_truth = dataset._format_text(first_item_wrapper)
    
    print("\n--- Input Info ---")
    print(f"Dataset: {first_item_wrapper['cfg'].name}")
    print(f"Question: {prompt_text}")
    print(f"Ground Truth: {ground_truth}")

    # 4. 构造推理输入
    messages = [
        {"role": "user", "content": [
            {"type": "image", "image": image},
            {"type": "text", "text": prompt_text}
        ]}
    ]
    
    text_prompt = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    
    inputs = processor(
        text=[text_prompt],
        images=[image],
        return_tensors="pt"
    )
    inputs = {k: v.to(args.device) for k, v in inputs.items()}

    # 5. 自回归生成
    print("\n--- Start Autoregressive Generation ---")
    with torch.no_grad():
        generated_ids = model.base_model.generate(
            **inputs,
            max_new_tokens=512, 
            use_cache=True, 
            do_sample=False, # 设为 False 以获得更稳定的结果
        )
    
    # 6. 解码并展示
    input_len = inputs['input_ids'].shape[1]
    output_ids = generated_ids[0][input_len:]
    output_text = processor.decode(output_ids, skip_special_tokens=False)
    
    print("\n--- Generated Output (Raw) ---")
    print(output_text)
    print("\n----------------------")
    
    # 检查隐式推理链是否生效
    has_prefix = config.extract_text_prefix.strip() in output_text
    has_stages = any(s.token in output_text for s in config.stages)
    
    if has_prefix or has_stages:
        print("✅ Latent Reasoning Chain detected in output.")
    else:
        print("❌ Model generated direct answer without latent chain.")

if __name__ == "__main__":
    main()