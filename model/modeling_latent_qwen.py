import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import Qwen2_5_VLForConditionalGeneration

class LatentReasoningQwen(nn.Module):
    def __init__(self, config_obj):
        super().__init__()
        self.config = config_obj
        self.len_tokenizer = None 

        # 加载 Base Model
        self.base_model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            config_obj.base_model,
            torch_dtype=torch.bfloat16,
            attn_implementation="sdpa" 
        )
        
        real_hidden_size = self.base_model.config.hidden_size
        
        # 冻结视觉塔
        if hasattr(self.base_model, "visual"):
            self.base_model.visual.requires_grad_(False)
            self.base_model.visual.eval()

        if config_obj.gradient_checkpointing:
            self.base_model.gradient_checkpointing_enable()

        # Projectors 用于 MSE 对齐
        self.projectors = nn.ModuleDict({
            stage.name: nn.Linear(real_hidden_size, stage.dim)
            for stage in config_obj.stages
        })
        
        # 缓存 Token IDs
        self.extract_token_id = config_obj.extract_token_id
        self.reasoning_token_ids = set([s.token_id for s in config_obj.stages])

    def forward(self, input_ids, attention_mask, labels, pixel_values, image_grid_thw, alignment_features=None):
        """
        VBC Forward:
        1. Pass 1: Standard (With Image)
        2. Pass 2: Blind (Image Masked)
        3. Compute Combined Loss
        """
        
        # === 1. Pass 1: Standard Forward (With Image) ===
        # 使用 standard causal mask
        outputs_v = self.base_model(
            input_ids=input_ids,
            attention_mask=attention_mask, 
            pixel_values=pixel_values,
            image_grid_thw=image_grid_thw,
            output_hidden_states=True,
            return_dict=True,
            use_cache=False 
        )
        logits_v = outputs_v.logits # [B, L, Vocab]
        
        # === 2. Pass 2: Blind Forward (Masking out Images) ===
        VISION_START_ID = 151652
        VISION_END_ID = 151653
        
        # 创建 blind mask (深拷贝)
        blind_attention_mask = attention_mask.clone()
        
        # 遍历 batch，将 Vision Token 区域的 mask 设为 0
        for b in range(input_ids.shape[0]):
            v_starts = (input_ids[b] == VISION_START_ID).nonzero(as_tuple=True)[0]
            v_ends = (input_ids[b] == VISION_END_ID).nonzero(as_tuple=True)[0]
            
            if len(v_starts) > 0:
                for start, end in zip(v_starts, v_ends):
                    blind_attention_mask[b, start:end+1] = 0
        
        # 执行 Blind Forward (no_grad 节省显存，且 VBC 通常 detach baseline)
        with torch.no_grad():
            outputs_blind = self.base_model(
                input_ids=input_ids,
                attention_mask=blind_attention_mask,
                pixel_values=None, # Blind 模式下无需图片张量
                image_grid_thw=None,
                output_hidden_states=False, 
                return_dict=True,
                use_cache=False
            )
            logits_blind = outputs_blind.logits.detach() 

        # === 3. 数据准备 (Shift for Next Token Prediction) ===
        shift_logits_v = logits_v[..., :-1, :].contiguous()
        shift_logits_blind = logits_blind[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()
        shift_input_ids = input_ids[..., 1:].contiguous()

        # Flatten
        flat_logits_v = shift_logits_v.view(-1, shift_logits_v.size(-1))
        flat_logits_blind = shift_logits_blind.view(-1, shift_logits_blind.size(-1))
        flat_labels = shift_labels.view(-1)
        flat_input_ids = shift_input_ids.view(-1)
        
        valid_mask = flat_labels != -100
        
        # === 4. 区分 Token 类型 (Special vs Answer) ===
        is_extract = (flat_input_ids == self.extract_token_id)
        is_reasoning = torch.zeros_like(flat_input_ids, dtype=torch.bool)
        for rid in self.reasoning_token_ids:
            is_reasoning |= (flat_input_ids == rid)
            
        # 注意: Text Prefix 被视为普通文本，归入 Answer 类型的 loss (即 Standard CE)
        # Special Tokens 只有那些 <ext>, <dino>
        is_special = (is_extract | is_reasoning) & valid_mask
        is_answer_or_text = (~is_special) & valid_mask
        
        # === 5. 计算损失 ===
        loss_fct = nn.CrossEntropyLoss(reduction='none') 
        ce_loss_all = loss_fct(flat_logits_v, flat_labels)
        
        # --- A. Latent Loss (Special Tokens) ---
        # 1. CE Loss: 强迫生成特殊 Token
        latent_ce_loss = torch.tensor(0.0, device=logits_v.device)
        if is_special.sum() > 0:
            latent_ce_loss = (ce_loss_all * is_special.float()).sum() / is_special.sum()
        
        # 2. MSE Loss: 特征对齐
        last_hidden_state = outputs_v.hidden_states[-1] # [B, L, H]
        total_mse_loss = torch.tensor(0.0, device=logits_v.device)
        mse_cnt = 0
        
        if alignment_features:
            for stage in self.config.stages:
                target_feat = alignment_features.get(stage.feature_key)
                if target_feat is None: continue
                
                stage_mask = (input_ids == stage.token_id)
                for b in range(input_ids.size(0)):
                    b_mask = stage_mask[b]
                    if b_mask.sum() > 0:
                        b_hidden = last_hidden_state[b][b_mask]
                        # Mean Pool
                        b_proj = self.projectors[stage.name](b_hidden.mean(dim=0, keepdim=True))
                        target = target_feat[b].unsqueeze(0).to(dtype=b_proj.dtype, device=b_proj.device)
                        total_mse_loss += F.mse_loss(b_proj, target)
                        mse_cnt += 1
                        
        if mse_cnt > 0:
            total_mse_loss /= mse_cnt

        loss_latent = latent_ce_loss + self.config.beta_mse * total_mse_loss

        # --- B. Answer Standard Loss ---
        ans_ce_loss = torch.tensor(0.0, device=logits_v.device)
        if is_answer_or_text.sum() > 0:
            ans_ce_loss = (ce_loss_all * is_answer_or_text.float()).sum() / is_answer_or_text.sum()

        # --- C. VBC Loss (Visual Bottleneck Contrast) ---
        loss_vbc = torch.tensor(0.0, device=logits_v.device)
        
        if is_answer_or_text.sum() > 0:
            # 1. 计算 Blind Probability P_blind(y_t)
            probs_blind = F.softmax(flat_logits_blind, dim=-1)
            p_blind_gt = probs_blind.gather(1, flat_labels.unsqueeze(1)).squeeze(1) # [N]
            
            # 2. 动态权重 w_t = max(0, margin - p_blind)
            weights = F.relu(self.config.vbc_margin - p_blind_gt)
            
            # 3. Contrastive Logits = Z_v - Z_blind
            logits_diff = flat_logits_v - flat_logits_blind
            
            # 4. CE(softmax(diff), y)
            loss_vbc_raw = loss_fct(logits_diff, flat_labels)
            
            loss_vbc = (loss_vbc_raw * weights * is_answer_or_text.float()).sum() / (is_answer_or_text.sum() + 1e-8)

        # === 6. Total Loss ===
        total_loss = (
            loss_latent + 
            ans_ce_loss + 
            self.config.lambda_vbc * loss_vbc
        )

        return {
            "loss": total_loss,
            "sft_loss": ans_ce_loss.detach(),
            "latent_ce": latent_ce_loss.detach(),
            "mse_loss": total_mse_loss.detach(),
            "vbc_loss": loss_vbc.detach()
        }