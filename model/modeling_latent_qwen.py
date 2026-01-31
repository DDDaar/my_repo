import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import Qwen2_5_VLForConditionalGeneration

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
        
        self.extract_token_id = config_obj.extract_token_id
        self.reasoning_token_ids = set([s.token_id for s in config_obj.stages])

    def forward(self, input_ids, attention_mask, labels, pixel_values, image_grid_thw, 
                alignment_features=None, answer_only_mask=None): # [新增参数]
        
        # === 1. Pass 1: Standard Forward (With Image) ===
        outputs_v = self.base_model(
            input_ids=input_ids,
            attention_mask=attention_mask, 
            pixel_values=pixel_values,
            image_grid_thw=image_grid_thw,
            output_hidden_states=True,
            return_dict=True,
            use_cache=False 
        )
        logits_v = outputs_v.logits
        
        # === 2. Pass 2: Blind Forward (Image Masked) ===
        VISION_START_ID = 151652
        VISION_END_ID = 151653
        
        blind_attention_mask = attention_mask.clone()
        
        # 动态分辨率下的 Blind Mask 生成
        for b in range(input_ids.shape[0]):
            v_starts = (input_ids[b] == VISION_START_ID).nonzero(as_tuple=True)[0]
            v_ends = (input_ids[b] == VISION_END_ID).nonzero(as_tuple=True)[0]
            
            if len(v_starts) > 0:
                for start, end in zip(v_starts, v_ends):
                    blind_attention_mask[b, start:end+1] = 0
        
        with torch.no_grad():
            outputs_blind = self.base_model(
                input_ids=input_ids,
                attention_mask=blind_attention_mask,
                pixel_values=None, 
                image_grid_thw=None,
                output_hidden_states=False, 
                return_dict=True,
                use_cache=False
            )
            logits_blind = outputs_blind.logits.detach() 

        # === 3. 数据 Shift (Next Token Prediction) ===
        shift_logits_v = logits_v[..., :-1, :].contiguous()
        shift_logits_blind = logits_blind[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()
        shift_input_ids = input_ids[..., 1:].contiguous()
        
        # [新增] Shift answer_only_mask 以对齐预测目标
        if answer_only_mask is not None:
            shift_answer_mask = answer_only_mask[..., 1:].contiguous().view(-1)
        else:
            shift_answer_mask = None

        flat_logits_v = shift_logits_v.view(-1, shift_logits_v.size(-1))
        flat_logits_blind = shift_logits_blind.view(-1, shift_logits_blind.size(-1))
        flat_labels = shift_labels.view(-1)
        flat_input_ids = shift_input_ids.view(-1)
        
        valid_mask = flat_labels != -100
        
        # === 4. 区域划分 ===
        is_extract = (flat_input_ids == self.extract_token_id)
        is_reasoning = torch.zeros_like(flat_input_ids, dtype=torch.bool)
        for rid in self.reasoning_token_ids:
            is_reasoning |= (flat_input_ids == rid)
            
        # 特殊 Token 区域
        is_special = (is_extract | is_reasoning) & valid_mask
        
        # 普通文本区域 (包括前缀废话 + 最终答案)
        is_answer_or_text = (~is_special) & valid_mask
        
        # === 5. 计算 Loss ===
        loss_fct = nn.CrossEntropyLoss(reduction='none') 
        ce_loss_all = loss_fct(flat_logits_v, flat_labels)
        
        # A. Latent Loss (Special Tokens 的 CE Loss)
        latent_ce_loss = torch.tensor(0.0, device=logits_v.device)
        if is_special.sum() > 0:
            latent_ce_loss = (ce_loss_all * is_special.float()).sum() / is_special.sum()
        
        # B. MSE Loss (Feature Alignment)
        last_hidden_state = outputs_v.hidden_states[-1] 
        total_mse_loss = torch.tensor(0.0, device=logits_v.device)
        mse_cnt = 0
        
        if alignment_features:
            for stage in self.config.stages:
                target_feat = alignment_features.get(stage.feature_key)
                if target_feat is None: continue
                
                # 确保该 batch 中确实存在该 token
                stage_mask = (input_ids == stage.token_id)
                if stage_mask.sum() == 0: continue

                for b in range(input_ids.size(0)):
                    b_mask = stage_mask[b]
                    if b_mask.sum() > 0:
                        b_hidden = last_hidden_state[b][b_mask]
                        # Mean Pool: 把多个 token 的隐层平均，去对齐一个 Global Feature
                        b_proj = self.projectors[stage.name](b_hidden.mean(dim=0, keepdim=True))
                        target = target_feat[b].unsqueeze(0).to(dtype=b_proj.dtype, device=b_proj.device)
                        total_mse_loss += F.mse_loss(b_proj, target)
                        mse_cnt += 1
        
        if mse_cnt > 0:
            total_mse_loss /= mse_cnt

        # C. Answer Standard Loss (SFT)
        # 计算所有非特殊 Token 的文本 Loss
        ans_ce_loss = torch.tensor(0.0, device=logits_v.device)
        if is_answer_or_text.sum() > 0:
            ans_ce_loss = (ce_loss_all * is_answer_or_text.float()).sum() / is_answer_or_text.sum()

        # D. VBC Loss (Visual Bottleneck Contrast)
        loss_vbc = torch.tensor(0.0, device=logits_v.device)
        
        # [关键逻辑] 确定 VBC 计算范围
        # 如果 scope 是 'answer' 且有 mask，则取交集：(是普通文本) AND (是最终答案)
        if self.config.vbc_target_scope == "answer" and shift_answer_mask is not None:
            vbc_target_mask = is_answer_or_text & shift_answer_mask
        else:
            # 否则 (scope='full')，所有普通文本都计算
            vbc_target_mask = is_answer_or_text
            
        if vbc_target_mask.sum() > 0:
            probs_blind = F.softmax(flat_logits_blind, dim=-1)
            p_blind_gt = probs_blind.gather(1, flat_labels.unsqueeze(1)).squeeze(1)
            
            # ReLU(Margin - P_blind)
            weights = F.relu(self.config.vbc_margin - p_blind_gt)
            
            logits_diff = flat_logits_v - flat_logits_blind
            loss_vbc_raw = loss_fct(logits_diff, flat_labels)
            
            loss_vbc = (loss_vbc_raw * weights * vbc_target_mask.float()).sum() / (vbc_target_mask.sum() + 1e-8)

        # === Total ===
        total_loss = (
            latent_ce_loss +  # [修正点] 之前为 loss_latent
            ans_ce_loss * self.config.alpha_sft + 
            total_mse_loss * self.config.beta_mse +
            loss_vbc * self.config.lambda_vbc
        )

        return {
            "loss": total_loss,
            "sft_loss": ans_ce_loss.detach(),
            "latent_ce": latent_ce_loss.detach(),
            "mse_loss": total_mse_loss.detach(),
            "vbc_loss": loss_vbc.detach()
        }