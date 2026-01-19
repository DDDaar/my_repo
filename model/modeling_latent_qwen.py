import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import Qwen2_5_VLForConditionalGeneration
import time

class LatentReasoningQwen(nn.Module):
    def __init__(self, config_obj):
        super().__init__()
        self.config = config_obj
        
        self.len_tokenizer = None  # Will be set externally after tokenizer is created

        self.base_model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            config_obj.base_model,
            torch_dtype=torch.bfloat16,
            attn_implementation="sdpa" 
        )
        

        # === 核心修复 1: 动态获取真实 Hidden Size ===
        real_hidden_size = self.base_model.config.hidden_size
        print(f"Real Model Hidden Size: {real_hidden_size} (Config says: {config_obj.hidden_size})")


        if hasattr(self.base_model, "visual"):
            print("❄️  Freezing Vision Encoder (NPU optimization)...")
            # 将视觉部分设为无需梯度，DeepSpeed 将自动跳过该部分的梯度计算
            self.base_model.visual.requires_grad_(False)
            # 确保视觉部分处于 eval 模式 (关闭 Dropout/BatchNorm 更新)
            self.base_model.visual.eval()

        # 开启梯度检查点 (如果配置要求)
        if config_obj.gradient_checkpointing:
            self.base_model.gradient_checkpointing_enable()

        self.projectors = nn.ModuleDict({
            stage.name: nn.Linear(real_hidden_size, stage.dim)
            for stage in config_obj.stages
        })
        
        self.pad_token_ids = set()
        if config_obj.extract_token_id:
            self.pad_token_ids.add(config_obj.extract_token_id)
        for s in config_obj.stages:
            self.pad_token_ids.add(s.token_id)

    def _create_custom_mask(self, input_ids, attention_mask):
        B, L = input_ids.shape
        device = input_ids.device
        min_dtype = torch.finfo(self.base_model.dtype).min
        
        causal_mask = torch.tril(torch.ones((L, L), device=device, dtype=torch.bool))
        padding_mask = attention_mask.view(B, 1, 1, L).bool()
        
        # Qwen2.5-VL 特殊 Token ID
        VISION_START_ID = 151652
        VISION_END_ID = 151653
        
        is_pad_token = torch.zeros_like(input_ids, dtype=torch.bool)
        for tid in self.pad_token_ids:
            is_pad_token |= (input_ids == tid)

        final_mask = torch.full((B, 1, L, L), min_dtype, device=device, dtype=self.base_model.dtype)
        
        for b in range(B):
            v_starts = (input_ids[b] == VISION_START_ID).nonzero(as_tuple=True)[0]
            v_ends = (input_ids[b] == VISION_END_ID).nonzero(as_tuple=True)[0]
            
            img_indices = []
            if len(v_starts) > 0 and len(v_ends) > 0:
                for start, end in zip(v_starts, v_ends):
                    img_indices.extend(range(start.item(), end.item() + 1))
            
            img_indices_tensor = torch.tensor(img_indices, device=device)
            
            current_causal = torch.zeros((L, L), device=device, dtype=self.base_model.dtype)
            current_causal.masked_fill_(~causal_mask, min_dtype)
            pad_row = attention_mask[b] == 0
            current_causal[:, pad_row] = min_dtype
            
            if len(img_indices) > 0:
                b_is_pad = is_pad_token[b]
                is_img_token = torch.zeros(L, device=device, dtype=torch.bool)
                is_img_token[img_indices_tensor] = True
                
                is_text_row = (~b_is_pad) & (~is_img_token)
                block_map = is_text_row.unsqueeze(1) & is_img_token.unsqueeze(0)
                current_causal.masked_fill_(block_map, min_dtype)

            final_mask[b, 0, :, :] = current_causal

        return final_mask

    def forward(
        self,
        input_ids,
        attention_mask,
        labels,
        pixel_values,
        image_grid_thw,
        alignment_features,
    ):
        # 1. 构建 Mask
        custom_mask = self._create_custom_mask(input_ids, attention_mask)
        
        # 2. Base Model Forward
        # [CRITICAL Fix]: 不传 labels 给 base_model，避免 shape invalid 报错
        outputs = self.base_model(
            input_ids=input_ids,
            attention_mask=custom_mask, 
            pixel_values=pixel_values,
            image_grid_thw=image_grid_thw,
            output_hidden_states=True,
            return_dict=True,
            use_cache=False # 训练必须 False
        )

        # 3. 手动计算 SFT Loss (CrossEntropy)
        # 解决 RuntimeError: shape invalid 的关键步骤
        logits = outputs.logits # [Batch, Seq, len(tokenizer)]
        
        # print("Logits shape:", logits.shape)

        time.sleep(1)

        # 偏移：用 t 预测 t+1
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()
        
        # print("Shifted Logits shape:", shift_logits.shape)
        # print("Shifted labels shape:", shift_labels.shape)

        # CrossEntropyLoss 自动忽略 index=-100
        loss_fct = nn.CrossEntropyLoss()

        # print(f'self.len_tokenizer={self.len_tokenizer}')
        self.len_tokenizer = self.len_tokenizer if self.len_tokenizer is not None else shift_logits.size(-1)

        sft_loss = loss_fct(
            shift_logits.view(-1, self.len_tokenizer), 
            shift_labels.view(-1)
        )

        # 4. 计算 MSE Loss (Latent Alignment)
        last_hidden_state = outputs.hidden_states[-1]
        total_mse_loss = 0.0
        mse_count = 0
        
        for stage in self.config.stages:
            target_feat = alignment_features.get(stage.feature_key)
            if target_feat is None: continue
                
            stage_mask = (input_ids == stage.token_id)
            if stage_mask.sum() == 0: continue

            batch_loss = 0.0
            valid_b = 0
            
            for b in range(input_ids.size(0)):
                b_mask = stage_mask[b]
                if b_mask.sum() > 0:
                    b_hidden = last_hidden_state[b][b_mask]
                    b_pooled = b_hidden.mean(dim=0, keepdim=True)
                    b_proj = self.projectors[stage.name](b_pooled)
                    
                    target = target_feat[b].unsqueeze(0).to(b_proj.dtype)
                    batch_loss += F.mse_loss(b_proj, target)
                    valid_b += 1
            
            if valid_b > 0:
                total_mse_loss += (batch_loss / valid_b)
                mse_count += 1

        avg_mse_loss = total_mse_loss / max(mse_count, 1)

        total_loss = (
            self.config.alpha_sft * sft_loss + 
            self.config.beta_mse * avg_mse_loss
        )

        return {
            "loss": total_loss,
            "sft_loss": sft_loss.detach(),
            "mse_loss": avg_mse_loss.detach()
        }