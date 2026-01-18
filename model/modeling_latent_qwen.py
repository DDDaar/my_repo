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
            attn_implementation="sdpa"
        )

        self.projectors = nn.ModuleDict({
            stage.name: nn.Linear(config_obj.hidden_size, stage.dim)
            for stage in config_obj.stages
        })

    def forward(
        self,
        input_ids,
        labels,
        pixel_values,
        image_grid_thw,
        alignment_features,
    ):
        """
        训练结构：
        1. VL 主干 forward（有梯度）
        2. latent reasoning（无梯度）
        3. latent token 插入
        4. CE + MSE loss
        """

        device = input_ids.device
        batch_size = input_ids.size(0)

        # ============================================================
        # 1️⃣ VL 主干 forward（有梯度）
        # ============================================================
        outputs = self.base_model(
            input_ids=input_ids,
            pixel_values=pixel_values,
            image_grid_thw=image_grid_thw,
            output_hidden_states=False,   # ❌ 不要开
            use_cache=False,              # ❌ 不要开
            return_dict=True,
        )

        hidden_states = outputs.last_hidden_state   # [B, L, H]

        # latent 起始位置
        start_indices = (input_ids == self.config.latent_start_id).int().argmax(dim=1)

        prefixes, suffixes = [], []
        for b in range(batch_size):
            idx = start_indices[b]
            prefixes.append(hidden_states[b:b+1, :idx+1])
            suffixes.append(hidden_states[b:b+1, idx+1:])

        # 初始 latent hidden（从 <LATENT> token 取）
        current_latent_h = torch.stack(
            [hidden_states[b, start_indices[b]] for b in range(batch_size)],
            dim=0
        ).unsqueeze(1)  # [B, 1, H]

        latent_tokens = [[] for _ in range(batch_size)]

        # ============================================================
        # 2️⃣ latent reasoning（无梯度，防 OOM）
        # ============================================================
        total_mse_loss = torch.tensor(0.0, device=device)
        mse_stage_cnt = 0

        with torch.no_grad():
            for stage in self.config.stages:
                target_feat = alignment_features[stage.feature_key]  # [B, D]
                stage_hiddens = []

                for _ in range(stage.steps):
                    step_out = self.base_model.model(
                        inputs_embeds=current_latent_h,
                        use_cache=False,
                        return_dict=True,
                    )

                    current_latent_h = step_out.last_hidden_state  # [B, 1, H]
                    stage_hiddens.append(current_latent_h)

                    for b in range(batch_size):
                        latent_tokens[b].append(
                            current_latent_h[b:b+1].detach()
                        )

                stage_cat = torch.cat(stage_hiddens, dim=1).mean(dim=1)  # [B, H]

                # 维度对齐
                resized = F.interpolate(
                    stage_cat.float().unsqueeze(1),
                    size=target_feat.shape[-1],
                    mode="linear",
                    align_corners=False
                ).squeeze(1)

                total_mse_loss += F.mse_loss(resized, target_feat.float())
                mse_stage_cnt += 1

        avg_mse_loss = total_mse_loss / max(mse_stage_cnt, 1)

        # ============================================================
        # 3️⃣ 拼接 latent token
        # ============================================================
        final_hidden = []
        for b in range(batch_size):
            seq = torch.cat(
                [prefixes[b]] + latent_tokens[b] + [suffixes[b]],
                dim=1
            )
            final_hidden.append(seq)

        combined_hidden = torch.cat(final_hidden, dim=0)

        # ============================================================
        # 4️⃣ SFT loss
        # ============================================================
        logits = self.base_model.lm_head(combined_hidden)

        total_latent_steps = sum(s.steps for s in self.config.stages)
        new_labels = self._expand_labels(labels, start_indices, total_latent_steps)

        shift_logits = logits[:, :-1].contiguous()
        shift_labels = new_labels[:, 1:].contiguous()

        loss_fct = nn.CrossEntropyLoss()
        sft_loss = loss_fct(
            shift_logits.view(-1, shift_logits.size(-1)),
            shift_labels.view(-1),
        )

        total_loss = (
            self.config.alpha_sft * sft_loss
            + self.config.beta_mse * avg_mse_loss
        )

        return {
            "loss": total_loss,
            "sft_loss": sft_loss.detach(),
            "mse_loss": avg_mse_loss.detach(),
        }

    def _expand_labels(self, labels, start_indices, num_new_tokens):
        device = labels.device
        new_labels = []

        for b in range(labels.size(0)):
            idx = start_indices[b]
            latent_pad = torch.full(
                (num_new_tokens,),
                -100,
                device=device,
                dtype=labels.dtype
            )
            new_labels.append(
                torch.cat([
                    labels[b, :idx+1],
                    latent_pad,
                    labels[b, idx+1:]
                ])
            )

        return nn.utils.rnn.pad_sequence(
            new_labels,
            batch_first=True,
            padding_value=-100
        )
