import torch
import deepspeed
from torch.utils.data import DataLoader, DistributedSampler
from transformers import Qwen2_5_VLProcessor
from configuration_latent import LatentConfig
from modeling_latent_qwen import LatentReasoningQwen
from dataset_latent import LatentReasoningDataset, collate_fn
import argparse


def parse_args():
	parser = argparse.ArgumentParser(description="Deepspeed Training")
	parser.add_argument("--local_rank", type=int, default=-1)
	parser = deepspeed.add_config_arguments(parser)
	return parser.parse_args()


def main():
	args = parse_args()

	# [cite_start]1. 加载配置 [cite: 1]
	config = LatentConfig.load("config.yaml")
	processor = Qwen2_5_VLProcessor.from_pretrained(config.base_model)

	# [cite_start]2. 注册 Token [cite: 2]
	special_tokens = [config.latent_start_token] + [f"{config.type1_base_token}{i}|>" for i in
	                                                range(config.type1_count)]
	processor.tokenizer.add_special_tokens({"additional_special_tokens": special_tokens})
	config.latent_start_id = processor.tokenizer.convert_tokens_to_ids(config.latent_start_token)

	# 3. 初始化模型
	model = LatentReasoningQwen(config)
	model.base_model.resize_token_embeddings(len(processor.tokenizer))

	# 4. 准备数据集 (此处以真实 ScienceQA 逻辑为例，匹配 preprocess_features.py)
	# 建议替换模拟数据为 load_dataset("derek-thomas/ScienceQA")
	train_data = [{"id": "0", "image_path": "test.jpg", "prompt": "Identify the object.", "answer": "This is a cat."}]
	dataset = LatentReasoningDataset(train_data, processor, config, "./data/aligned_features")

	sampler = DistributedSampler(dataset)
	dataloader = DataLoader(
		dataset,
		batch_size=config.batch_size,
		collate_fn=collate_fn,
		sampler=sampler
	)

	# 5. DeepSpeed 初始化
	# [cite_start]自动处理 Optimizer, LR Scheduler 和 ZeRO 状态 [cite: 2]
	model_engine, optimizer, _, _ = deepspeed.initialize(
		args=args,
		model=model,
		model_parameters=model.parameters(),
		config="ds_config.json"
	)

	model_engine.train()
	for epoch in range(10):  # 示例 epoch
		sampler.set_epoch(epoch)
		for batch in dataloader:
			# [cite_start]将数据移动到设备 [cite: 2]
			input_ids = batch["input_ids"].to(model_engine.device)
			labels = batch["labels"].to(model_engine.device)
			pixel_values = batch["pixel_values"].to(model_engine.device, dtype=torch.bfloat16)
			image_grid_thw = batch["image_grid_thw"].to(model_engine.device)
			alignment_features = {k: v.to(model_engine.device, dtype=torch.bfloat16)
			                      for k, v in batch["alignment_features"].items()}

			# 前向传播
			outputs = model_engine(
				input_ids=input_ids,
				labels=labels,
				pixel_values=pixel_values,
				image_grid_thw=image_grid_thw,
				alignment_features=alignment_features
			)

			loss = outputs["loss"]

			# [cite_start]反向传播 [cite: 2]
			model_engine.backward(loss)
			model_engine.step()

			if args.local_rank <= 0:
				print(
					f"Loss: {loss.item():.4f} (SFT: {outputs['loss_sft'].item():.4f}, MSE: {outputs['loss_mse'].item():.4f})")


if __name__ == "__main__":
	main()