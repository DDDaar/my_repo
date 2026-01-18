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

    def forward(self, input_ids, labels, pixel_values, image_grid_thw, alignment_features):
        batch_size = input_ids.shape[0]
        device = input_ids.device

        # 1. 基础 Pass
        outputs = self.base_model(
            input_ids=input_ids,
            pixel_values=pixel_values,
            image_grid_thw=image_grid_thw,
            output_hidden_states=True,
            use_cache=True,
            return_dict=True
        )
        
        all_hidden_states = outputs.hidden_states[-1] 
        past_key_values = outputs.past_key_values

        start_indices = (input_ids == self.config.latent_start_id).int().argmax(dim=1)

        prefixes = []
        suffixes = []
        for b in range(batch_size):
            idx = start_indices[b]
            prefixes.append(all_hidden_states[b:b+1, :idx+1, :])
            suffixes.append(all_hidden_states[b:b+1, idx+1:, :])

        total_mse_loss = torch.tensor(0.0, device=device, dtype=torch.float32)
        mse_stages = 0
        latent_tokens_list = [[] for _ in range(batch_size)]

        current_latent_h = []
        for b in range(batch_size):
            idx = start_indices[b]
            current_latent_h.append(all_hidden_states[b:b+1, idx:idx+1, :])
        current_latent_h = torch.cat(current_latent_h, dim=0)

        for stage in self.config.stages:
            target_feat = alignment_features[stage.feature_key] 
            target_dim = target_feat.shape[-1]
            stage_step_hiddens = []

            for _ in range(stage.steps):
                # 增量推理：此时不需要 pixel_values，信息已在 past_key_values 中
                step_outputs = self.base_model.model(
                    inputs_embeds=current_latent_h,
                    past_key_values=past_key_values,
                    use_cache=True,
                    return_dict=True
                )
                
                current_latent_h = step_outputs.last_hidden_state
                past_key_values = step_outputs.past_key_values
                
                stage_step_hiddens.append(current_latent_h)
                for b in range(batch_size):
                    latent_tokens_list[b].append(current_latent_h[b:b+1, :, :])

            stage_h_cat = torch.cat(stage_step_hiddens, dim=1)
            stage_avg_hidden = stage_h_cat.mean(dim=1)

            resized = F.interpolate(
                stage_avg_hidden.float().unsqueeze(1), 
                size=target_dim,
                mode="linear",
                align_corners=False
            ).squeeze(1)

            total_mse_loss += F.mse_loss(resized, target_feat.float())
            mse_stages += 1

        avg_mse_loss = total_mse_loss / mse_stages if mse_stages > 0 else 0.0

        final_embeddings = []
        for b in range(batch_size):
            sample_full_seq = torch.cat([prefixes[b]] + latent_tokens_list[b] + [suffixes[b]], dim=1)
            final_embeddings.append(sample_full_seq)

        combined_hidden = torch.cat(final_embeddings, dim=0)
        logits = self.base_model.lm_head(combined_hidden)

        total_latent_steps = sum(stage.steps for stage in self.config.stages)
        new_labels = self._expand_labels(labels, start_indices, total_latent_steps)

        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = new_labels[..., 1:].contiguous()

        loss_fct = nn.CrossEntropyLoss()
        sft_loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))

        total_loss = self.config.alpha_sft * sft_loss + self.config.beta_mse * avg_mse_loss
        
        return {
            "loss": total_loss, 
            "sft_loss": sft_loss, 
            "mse_loss": avg_mse_loss
        }

    def _expand_labels(self, labels, start_indices, num_new_tokens):
        batch_size, seq_len = labels.shape
        device = labels.device
        new_labels_list = []
        for b in range(batch_size):
            idx = start_indices[b]
            prefix_l = labels[b, :idx+1]
            suffix_l = labels[b, idx+1:]
            latent_l = torch.full((num_new_tokens,), -100, device=device, dtype=labels.dtype)
            new_labels_list.append(torch.cat([prefix_l, latent_l, suffix_l]))
        
        # 统一长度进行 padding，防止 batch 内序列长度不一
        return torch.nn.utils.rnn.pad_sequence(new_labels_list, batch_first=True, padding_value=-100)