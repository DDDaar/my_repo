import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import Qwen2_5_VLForConditionalGeneration
import time

class LatentReasoningQwen(nn.Module):
    def __init__(self, config_obj):
        super().__init__()
        self.config = config_obj
        self.len_tokenizer = None 

        self.base_model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            config_obj.base_model,
            torch_dtype=torch.bfloat16,
            attn_implementation="sdpa" 
        )
        
        real_hidden_size = self.base_model.config.hidden_size
        
        if hasattr(self.base_model, "visual"):
            self.base_model.visual.requires_grad_(False)
            self.base_model.visual.eval()

        if config_obj.gradient_checkpointing:
            self.base_model.gradient_checkpointing_enable()

        self.projectors = nn.ModuleDict({
            stage.name: nn.Linear(real_hidden_size, stage.dim)
            for stage in config_obj.stages
        })
        
        # 缓存 Token ID 用于 Mask 生成
        self.extract_token_id = config_obj.extract_token_id
        self.reasoning_token_ids = set([s.token_id for s in config_obj.stages])
        
        # 解析可见性集合
        self.visible_conf = set(config_obj.image_visible_to)

    def _create_custom_mask(self, input_ids, attention_mask):
        """
        生成注意力掩码，控制 Question/Extract/Reasoning/Answer 对 Image 的可见性。
        """
        B, L = input_ids.shape
        device = input_ids.device
        min_dtype = torch.finfo(self.base_model.dtype).min
        
        # 1. 基础 Causal Mask (Lower Triangular)
        causal_mask = torch.tril(torch.ones((L, L), device=device, dtype=torch.bool))
        
        # 初始化最终 Mask: [B, 1, L, L] 用于广播
        final_mask = torch.full((B, 1, L, L), min_dtype, device=device, dtype=self.base_model.dtype)
        
        VISION_START_ID = 151652
        VISION_END_ID = 151653
        
        for b in range(B):
            # 基础 Causal
            current_causal = torch.zeros((L, L), device=device, dtype=self.base_model.dtype)
            current_causal.masked_fill_(~causal_mask, min_dtype)
            
            # Padding 处理 (attention_mask 为 0 的列不可见)
            pad_col = attention_mask[b] == 0
            current_causal[:, pad_col] = min_dtype

            # 寻找图像 Token 区域
            v_starts = (input_ids[b] == VISION_START_ID).nonzero(as_tuple=True)[0]
            v_ends = (input_ids[b] == VISION_END_ID).nonzero(as_tuple=True)[0]
            
            img_indices = []
            if len(v_starts) > 0:
                for start, end in zip(v_starts, v_ends):
                    img_indices.extend(range(start.item(), end.item() + 1))
            
            # 如果没有图片，直接用基础 Causal
            if not img_indices:
                final_mask[b, 0, :, :] = current_causal
                continue

            img_indices_tensor = torch.tensor(img_indices, device=device)
            is_img_token = torch.zeros(L, device=device, dtype=torch.bool)
            is_img_token[img_indices_tensor] = True

            # === 划分 Token 类型 ===
            # 类型定义: 0: Question, 1: Extract, 2: Reasoning, 3: Answer
            token_types = torch.zeros(L, device=device, dtype=torch.int) 
            
            # 标记 Extract
            token_types[input_ids[b] == self.extract_token_id] = 1
            
            # 标记 Reasoning
            for rid in self.reasoning_token_ids:
                token_types[input_ids[b] == rid] = 2
            
            # 标记 Answer: 位于最后一个 Reasoning/Extract 之后的所有 Token
            special_indices = torch.where((token_types == 1) | (token_types == 2))[0]
            if len(special_indices) > 0:
                last_special = special_indices.max().item()
                if last_special + 1 < L:
                    token_types[last_special + 1:] = 3
            
            # === 构建可见性向量 ===
            can_see_image = torch.zeros(L, device=device, dtype=torch.bool)
            
            # 规则 1: Question 默认必须看图 (Type 0)
            can_see_image[token_types == 0] = True
            
            # 规则 2: Extract Token (Type 1)
            if "extract_token" in self.visible_conf:
                can_see_image[token_types == 1] = True
                
            # 规则 3: Reasoning Token (Type 2)
            if "reasoning_tokens" in self.visible_conf:
                can_see_image[token_types == 2] = True
                
            # 规则 4: Answer (Type 3)
            if "answer" in self.visible_conf:
                can_see_image[token_types == 3] = True
            
            # 规则 5: Image Token 自身互看
            can_see_image[is_img_token] = True
            
            # === 应用遮罩 ===
            # Mask 逻辑: 如果某行 (Row) 不能看图，则在该行对应 Image 列 (Col) 填 -inf
            
            # mask_rows[i] = True 意味着第 i 个 token 禁止看图
            mask_rows = ~can_see_image
            
            # 构造 Block Map: [L, 1] & [1, L] -> [L, L]
            block_map = mask_rows.unsqueeze(1) & is_img_token.unsqueeze(0)
            
            current_causal.masked_fill_(block_map, min_dtype)
            final_mask[b, 0, :, :] = current_causal

        return final_mask

    def forward(self, input_ids, attention_mask, labels, pixel_values, image_grid_thw, alignment_features=None):
        # 1. 构建 Mask (包含 Visibility 控制)
        custom_mask = self._create_custom_mask(input_ids, attention_mask)
        
        # 2. Base Model Forward
        outputs = self.base_model(
            input_ids=input_ids,
            attention_mask=custom_mask, 
            pixel_values=pixel_values,
            image_grid_thw=image_grid_thw,
            output_hidden_states=True,
            return_dict=True,
            use_cache=False 
        )

        logits = outputs.logits 
        
        # 3. SFT Loss
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()
        
        loss_fct = nn.CrossEntropyLoss()
        
        if self.len_tokenizer is None:
            self.len_tokenizer = shift_logits.size(-1)

        sft_loss = loss_fct(
            shift_logits.view(-1, self.len_tokenizer), 
            shift_labels.view(-1)
        )

        # 4. MSE Loss (Latent Alignment)
        avg_mse_loss = torch.tensor(0.0, device=logits.device)
        
        if alignment_features and len(alignment_features) > 0:
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
                        # 对该 Stage 的所有 Token 取平均后做 MSE
                        b_pooled = b_hidden.mean(dim=0, keepdim=True)
                        b_proj = self.projectors[stage.name](b_pooled)
                        
                        target = target_feat[b].unsqueeze(0).to(dtype=b_proj.dtype, device=b_proj.device)
                        batch_loss += F.mse_loss(b_proj, target)
                        valid_b += 1
                
                if valid_b > 0:
                    total_mse_loss += (batch_loss / valid_b)
                    mse_count += 1
            
            if mse_count > 0:
                avg_mse_loss = total_mse_loss / mse_count

        total_loss = (
            self.config.alpha_sft * sft_loss + 
            self.config.beta_mse * avg_mse_loss
        )

        return {
            "loss": total_loss,
            "sft_loss": sft_loss.detach(),
            "mse_loss": avg_mse_loss.detach()
        }