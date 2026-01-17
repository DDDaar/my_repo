import torch
import torch.nn as nn
from transformers import Qwen2_5_VLForConditionalGeneration


class LatentReasoningQwen(nn.Module):
	def __init__(self, config_obj):
		super().__init__()
		self.config = config_obj
		self.base_model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
			config_obj.base_model, torch_dtype=torch.bfloat16
		)
		# 为各阶段创建对齐投影头
		self.projectors = nn.ModuleDict({
			stage.name: nn.Linear(config_obj.hidden_size, stage.dim)
			for stage in config_obj.stages
		})

	def create_custom_mask(self, input_ids, labels, image_token_id):
		batch_size, seq_len = input_ids.shape
		# 默认使用因果掩码 (Causal Mask)
		mask = torch.tril(torch.ones((seq_len, seq_len), device=input_ids.device)).view(1, 1, seq_len, seq_len)

		for b in range(batch_size):
			img_indices = (input_ids[b] == image_token_id).nonzero(as_tuple=True)[0]
			latent_idx = (input_ids[b] == self.config.latent_start_id).nonzero(as_tuple=True)[0]
			answer_indices = (labels[b] != -100).nonzero(as_tuple=True)[0]

			# 根据配置屏蔽可见性
			if not self.config.vis_mask['latent_sees_image'] and len(latent_idx) > 0:
				mask[b, 0, latent_idx, img_indices] = 0

			if not self.config.vis_mask['answer_sees_image'] and len(answer_indices) > 0:
				for a_idx in answer_indices:
					mask[b, 0, a_idx, img_indices] = 0

		return (1.0 - mask) * torch.finfo(torch.bfloat16).min

	def forward(self, input_ids, labels, pixel_values, image_grid_thw, alignment_features):
		batch_size = input_ids.shape[0]

		# 1. 构造自定义掩码并执行初始编码
		# 注意：Qwen2.5-VL 内部会自动处理图像 token 展开，此处简化处理，实际需对应展开后的索引
		custom_attn_mask = self.create_custom_mask(input_ids, labels, image_token_id=151655)  # 假设的图像 token id

		outputs = self.base_model(
			input_ids=input_ids,
			pixel_values=pixel_values,
			image_grid_thw=image_grid_thw,
			output_hidden_states=True,
			return_dict=True
			# attention_mask=custom_attn_mask # 需确保基类支持传入自定义 4D mask
		)

		# 2. 定位 Latent 起点并提取初始隐藏状态
		mask_latent = (input_ids == self.config.latent_start_id)
		start_indices = mask_latent.int().argmax(dim=1)
		current_hidden = outputs.hidden_states[-1][torch.arange(batch_size), start_indices, :].unsqueeze(1)

		# 3. 潜在推理循环 (Latent Reasoning Loop)
		total_mse_loss = 0.0
		mse_steps = 0
		for stage in self.config.stages:
			target_feat = alignment_features[stage.feature_key]
			proj_layer = self.projectors[stage.name]
			for _ in range(stage.steps):
				# 将状态输入 Transformer 层进行一次迭代思考
				latent_out = self.base_model.model(inputs_embeds=current_hidden, use_cache=False)
				current_hidden = latent_out.last_hidden_state
				# 计算 MSE Loss
				proj_feat = proj_layer(current_hidden.squeeze(1))
				total_mse_loss += nn.functional.mse_loss(proj_feat.float(), target_feat.float())
				mse_steps += 1

		avg_mse_loss = total_mse_loss / mse_steps if mse_steps > 0 else 0.0

		# 4. 状态注入：将思考后的状态放回原序列位置
		modified_hidden = outputs.hidden_states[-1].clone()
		modified_hidden[torch.arange(batch_size), start_indices, :] = current_hidden.squeeze(1)

		# 5. 答案生成预测：基于修改后的隐藏状态继续回归
		logits = self.base_model.lm_head(modified_hidden)

		# 6. SFT Loss 计算 (仅计算答案部分)
		shift_logits = logits[..., :-1, :].contiguous()
		shift_labels = labels[..., 1:].contiguous()
		loss_fct = nn.CrossEntropyLoss(ignore_index=-100)
		loss_sft = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))

		return {
			"loss": self.config.alpha_sft * loss_sft + self.config.beta_mse * avg_mse_loss,
			"loss_sft": loss_sft,
			"loss_mse": avg_mse_loss
		}