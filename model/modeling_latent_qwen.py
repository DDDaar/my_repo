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
        # 投影头保持不变
        self.projectors = nn.ModuleDict({
            stage.name: nn.Linear(config_obj.hidden_size, stage.dim)
            for stage in config_obj.stages
        })

    def forward(self, input_ids, labels, pixel_values, image_grid_thw, alignment_features):
        batch_size = input_ids.shape[0]

        # 1. 初始编码：获取所有原始 Token 的隐藏状态
        outputs = self.base_model(
            input_ids=input_ids,
            pixel_values=pixel_values,
            image_grid_thw=image_grid_thw,
            output_hidden_states=True,
            return_dict=True
        )
        
        # 原始隐藏状态 [B, Seq_Len, Hidden_Dim]
        all_hidden_states = outputs.hidden_states[-1]
        
        # 2. 定位 latent_start_token
        start_indices = (input_ids == self.config.latent_start_id).int().argmax(dim=1)
        
        # 拆分序列：前缀（包含 latent_start）和 后缀（Answer部分）
        # prefix: [B, Start_Idx + 1, H]
        # suffix: [B, Seq_Len - Start_Idx - 1, H]
        prefixes = []
        suffixes = []
        for b in range(batch_size):
            idx = start_indices[b]
            prefixes.append(all_hidden_states[b:b+1, :idx+1, :])
            suffixes.append(all_hidden_states[b:b+1, idx+1:, :])

        # 3. 动态推理循环（序列扩展）
        total_mse_loss = 0.0
        mse_steps = 0
        
        # 存储生成的推理隐向量
        latent_tokens_list = [[] for _ in range(batch_size)]

        for stage in self.config.stages:
            target_feat = alignment_features[stage.feature_key]
            proj_layer = self.projectors[stage.name]
            
            for _ in range(stage.steps):
                for b in range(batch_size):
                    # 当前该样本的完整序列 = 前缀 + 已产生的推理Token
                    current_seq = torch.cat([prefixes[b]] + latent_tokens_list[b], dim=1)
                    
                    # 将当前序列喂入 Transformer 层，只取最后一个位置的输出作为新产生的推理 Token
                    # 注意：这里为了效率通常只运行一层或使用 cache，但为了逻辑严谨我们运行 base_model 的 model 部分
                    # 提示：实际生产中建议只通过最后的 block 以节省开销
                    latent_out = self.base_model.model(
                        inputs_embeds=current_seq, 
                        use_cache=False
                    ).last_hidden_state[:, -1:, :] # 提取新生成的末尾向量 [1, 1, H]
                    
                    # 计算此步产生的向量与视觉特征的 MSE
                    proj_feat = proj_layer(latent_out.squeeze(1))
                    total_mse_loss += nn.functional.mse_loss(proj_feat.float(), target_feat[b:b+1].float())
                    mse_steps += 1
                    
                    # 【核心修改】：Append 到列表中，增加序列长度
                    latent_tokens_list[b].append(latent_out)

        avg_mse_loss = total_mse_loss / mse_steps if mse_steps > 0 else 0.0

        # 4. 序列重组：[Prefix] + [Latent_Tokens...] + [Suffix]
        final_embeddings = []
        for b in range(batch_size):
            sample_full_seq = torch.cat([prefixes[b]] + latent_tokens_list[b] + [suffixes[b]], dim=1)
            final_embeddings.append(sample_full_seq)
        
        # 拼接成 Batch [B, New_Seq_Len, H]
        # 注意：因为增加了推理 Token，这里的 New_Seq_Len = 原长度 + 总推理步数
        combined_hidden = torch.cat(final_embeddings, dim=0)

        # 5. 最终预测与 Label 对齐
        # 关键：由于序列变长了，原本的 labels 也需要对应插入 -100 占位符
        logits = self.base_model.lm_head(combined_hidden)
        
        # 构造新 Labels：在 latent_start 后面插入 N 个 -100
        total_latent_steps = sum(stage.steps for stage in self.config.stages)
        new_labels = self._expand_labels(labels, start_indices, total_latent_steps)

        # 计算 SFT Loss
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = new_labels[..., 1:].contiguous()
        
        loss_fct = nn.CrossEntropyLoss()
        sft_loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))

        # 最终组合 Loss
        total_loss = self.config.alpha_sft * sft_loss + self.config.beta_mse * avg_mse_loss

        return {"loss": total_loss, "sft_loss": sft_loss, "mse_loss": avg_mse_loss}

    def _expand_labels(self, labels, start_indices, num_new_tokens):
        """
        在 labels 的 latent_start 位置后插入 num_new_tokens 个 -100
        """
        batch_size, seq_len = labels.shape
        new_labels = []
        for b in range(batch_size):
            idx = start_indices[b]
            # [前缀标签] + [-100 * N] + [后缀标签]
            prefix_l = labels[b, :idx+1]
            suffix_l = labels[b, idx+1:]
            latent_l = torch.full((num_new_tokens,), -100, device=labels.device, dtype=labels.dtype)
            new_labels.append(torch.cat([prefix_l, latent_l, suffix_l]))
        return torch.stack(new_labels)