import os
import torch
import argparse
from transformers import Qwen2_5_VLProcessor
from model.modeling_latent_qwen import LatentReasoningQwen
from config.configuration_latent import LatentConfig
from PIL import Image

def load_image(image_path):
    if not os.path.exists(image_path):
        return Image.new('RGB', (224, 224), (0, 0, 0))
    return Image.open(image_path).convert("RGB")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, required=True, help="HF format checkpoint path")
    parser.add_argument("--config", type=str, default="./config/config.yaml")
    parser.add_argument("--image_path", type=str, default="test.jpg")
    parser.add_argument("--question", type=str, default="Describe the image in detail.")
    parser.add_argument("--device", type=str, default="cuda")
    args = parser.parse_args()

    # 1. 加载配置
    print(f"Loading Config from {args.config}...")
    config = LatentConfig.load(args.config)
    
    # 2. 加载 Processor 和 Model
    print(f"Loading Model from {args.checkpoint}...")
    processor = Qwen2_5_VLProcessor.from_pretrained(args.checkpoint)
    
    # 确保 config ID 与 tokenizer 同步
    config.extract_token_id = processor.tokenizer.convert_tokens_to_ids(config.extract_token)
    for stage in config.stages:
        stage.token_id = processor.tokenizer.convert_tokens_to_ids(stage.token)
        
    model = LatentReasoningQwen(config)
    model.base_model = model.base_model.from_pretrained(args.checkpoint, torch_dtype=torch.bfloat16)
    
    # 加载 Projector 权重
    proj_path = os.path.join(args.checkpoint, "projectors.bin")
    if os.path.exists(proj_path):
        model.projectors.load_state_dict(torch.load(proj_path, map_location="cpu"))
        print("Projectors loaded.")
    
    model.to(args.device)
    model.eval()

    # 3. 构造输入 (仅 User Prompt)
    messages = [
        {"role": "user", "content": [
            {"type": "image", "image": args.image_path},
            {"type": "text", "text": args.question}
        ]}
    ]
    
    text_prompt = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    image_obj = load_image(args.image_path)
    
    inputs = processor(
        text=[text_prompt],
        images=[image_obj],
        return_tensors="pt"
    )
    inputs = {k: v.to(args.device) for k, v in inputs.items()}
    if "image_grid_thw" in inputs:
        inputs["image_grid_thw"] = inputs["image_grid_thw"].to(args.device)

    # 4. 自回归生成
    print("\n--- Start Autoregressive Generation ---")
    with torch.no_grad():
        # max_new_tokens 设置大一点，因为包含了 Reasoning Chain
        generated_ids = model.base_model.generate(
            **inputs,
            max_new_tokens=512, 
            use_cache=True, 
            do_sample=True,
            temperature=0.7,
            top_p=0.9
        )
    
    # 5. 解码并展示
    input_len = inputs['input_ids'].shape[1]
    output_ids = generated_ids[0][input_len:]
    
    # 不跳过特殊 token，以便观察是否生成了 <ext> 等
    output_text = processor.decode(output_ids, skip_special_tokens=False)
    
    print("\n--- Generated Output ---")
    print(output_text)
    print("\n----------------------")
    
    # 验证是否按照预期生成
    if config.extract_text_prefix.strip() in output_text:
        print("✅ Extract Prefix detected.")
    else:
        print("❌ Extract Prefix missing (Model might need more training).")

if __name__ == "__main__":
    main()