import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import Qwen2_5_VLForConditionalGeneration

class LatentReasoningQwen(nn.Module):
    def __init__(self, config_obj):
        super().__init__()
        self.config = config_obj
        self.base_model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            config_obj.base_model, 
            torch_dtype=torch.bfloat16,
            attn_implementation="flash_attention_2"  # 建议开启以进一步省显存
        )
        
        # 投影头（根据配置保留，若不使用可设为 requires_grad=False）
        self.projectors = nn.ModuleDict({
            stage.name: nn.Linear(config_obj.hidden_size, stage.dim)
            for stage in config_obj.stages
        })

    def forward(self, input_ids, labels, pixel_values, image_grid_thw, alignment_features):
        batch_size = input_ids.shape[0]
        device = input_ids.device
        dtype = self.base_model.dtype

        # 1. 基础 Pass：处理图像和原始文本输入
        # 开启 use_cache 以获取初始 KV Cache
        outputs = self.base_model(
            input_ids=input_ids,
            pixel_values=pixel_values,
            image_grid_thw=image_grid_thw,
            output_hidden_states=True,
            use_cache=True,
            return_dict=True
        )
        
        # 获取第一阶段结束后的隐藏状态和缓存
        all_hidden_states = outputs.hidden_states[-1] # [B, L, H]
        past_key_values = outputs.past_key_values

        # 2. 定位 latent_start_token 位置
        # 假设所有样本在同一 batch 中 latent_start_id 出现位置相同或我们处理增量
        start_indices = (input_ids == self.config.latent_start_id).int().argmax(dim=1)

        # 提取前缀和后缀的隐藏状态（用于最后拼接 SFT 序列）
        prefixes = []
        suffixes = []
        for b in range(batch_size):
            idx = start_indices[b]
            prefixes.append(all_hidden_states[b:b+1, :idx+1, :])
            suffixes.append(all_hidden_states[b:b+1, idx+1:, :])

        # 3. 动态推理循环（KV Cache 模式）
        total_mse_loss = torch.tensor(0.0, device=device, dtype=torch.float32)
        mse_stages = 0
        latent_tokens_list = [[] for _ in range(batch_size)]

        # 获取当前用于推理的最后一个隐藏状态作为“种子”
        # 我们取 latent_start_token 对应的输出
        current_latent_h = []
        for b in range(batch_size):
            idx = start_indices[b]
            current_latent_h.append(all_hidden_states[b:b+1, idx:idx+1, :])
        current_latent_h = torch.cat(current_latent_h, dim=0) # [B, 1, H]

        for stage in self.config.stages:
            target_feat = alignment_features[stage.feature_key] # [B, D_target]
            target_dim = target_feat.shape[-1]
            stage_step_hiddens = []

            for _ in range(stage.steps):
                # 使用增量隐藏状态进行推理
                # 注意：Qwen2_5_VL 的 model 接受 inputs_embeds
                step_outputs = self.base_model.model(
                    inputs_embeds=current_latent_h,
                    past_key_values=past_key_values,
                    use_cache=True,
                    return_dict=True
                )
                
                # 更新隐藏状态和 Cache
                current_latent_h = step_outputs.last_hidden_state # [B, 1, H]
                past_key_values = step_outputs.past_key_values
                
                # 记录这一步的输出
                stage_step_hiddens.append(current_latent_h)
                for b in range(batch_size):
                    latent_tokens_list[b].append(current_latent_h[b:b+1, :, :])

            # --- Stage 内 MSE 计算 ---
            # 聚合当前 Stage 的所有步骤 [B, steps, H]
            stage_h_cat = torch.cat(stage_step_hiddens, dim=1)
            stage_avg_hidden = stage_h_cat.mean(dim=1) # [B, H]

            # 1D 插值对齐维度
            resized = F.interpolate(
                stage_avg_hidden.float().unsqueeze(1), # [B, 1, H]
                size=target_dim,
                mode="linear",
                align_corners=False
            ).squeeze(1) # [B, target_dim]

            total_mse_loss += F.mse_loss(resized, target_feat.float())
            mse_stages += 1

        avg_mse_loss = total_mse_loss / mse_stages if mse_stages > 0 else 0.0

        # 4. 序列重组（构建完整的 SFT 训练序列）
        final_embeddings = []
        for b in range(batch_size):
            # 拼接: Prefix (含 Start Token) + All Latent Steps + Suffix
            sample_full_seq = torch.cat([prefixes[b]] + latent_tokens_list[b] + [suffixes[b]], dim=1)
            final_embeddings.append(sample_full_seq)

        combined_hidden = torch.cat(final_embeddings, dim=0)

        # 5. 最终预测与 Label 对齐
        logits = self.base_model.lm_head(combined_hidden)

        total_latent_steps = sum(stage.steps for stage in self.config.stages)
        new_labels = self._expand_labels(labels, start_indices, total_latent_steps)

        # 标准 Causal LM Loss 计算
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = new_labels[..., 1:].contiguous()

        loss_fct = nn.CrossEntropyLoss()
        sft_loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))

        # 总损失平衡
        total_loss = self.config.alpha_sft * sft_loss + self.config.beta_mse * avg_mse_loss
        
        return {
            "loss": total_loss, 
            "sft_loss": sft_loss, 
            "mse_loss": avg_mse_loss
        }

    def _expand_labels(self, labels, start_indices, num_new_tokens):
        """
        在 labels 的 latent_start_token 之后插入 -100，使模型不学习推理隐向量的生成
        """
        batch_size, seq_len = labels.shape
        device = labels.device
        new_labels = []
        for b in range(batch_size):
            idx = start_indices[b]
            prefix_l = labels[b, :idx+1]
            suffix_l = labels[b, idx+1:]
            # 推理位填充 -100，不计算 CrossEntropy
            latent_l = torch.full((num_new_tokens,), -100, device=device, dtype=labels.dtype)
            new_labels.append(torch.cat([prefix_l, latent_l, suffix_l]))
        return torch.stack(new_labels)