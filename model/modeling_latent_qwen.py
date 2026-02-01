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
        # 仅追踪特征 Token 用于 MSE 对齐
        self.feature_token_ids = set([s.token_id for s in config_obj.stages])

    def forward(self, input_ids, attention_mask, labels, pixel_values, image_grid_thw, 
                alignment_features=None, answer_only_mask=None, has_visual_features=None): 
        
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
        
        # === 2. Pass 2: Blind Forward (Image Masked for VBC) ===
        # 只有存在视觉特征的样本才进行 Blind Forward，否则浪费计算
        # 但为了 Batch 并行，我们必须对整个 Batch forward，然后用 mask 过滤 loss
        # 这里进行一个优化：如果有 valid_visual_samples，则计算，否则跳过
        
        do_vbc = self.config.lambda_vbc > 0
        if has_visual_features is not None and has_visual_features.sum() == 0:
            do_vbc = False # 全是纯文本，不跑 VBC
            
        logits_blind = None
        if do_vbc:
            VISION_START_ID = 151652
            VISION_END_ID = 151653
            
            blind_attention_mask = attention_mask.clone()
            for b in range(input_ids.shape[0]):
                # 如果这个样本本来就没有视觉特征，mask 不 mask 没区别，不影响结果，只影响 loss mask
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
        shift_labels = labels[..., 1:].contiguous()
        shift_input_ids = input_ids[..., 1:].contiguous()
        
        if answer_only_mask is not None:
            shift_answer_mask = answer_only_mask[..., 1:].contiguous().view(-1)
        else:
            shift_answer_mask = None

        flat_logits_v = shift_logits_v.view(-1, shift_logits_v.size(-1))
        flat_labels = shift_labels.view(-1)
        flat_input_ids = shift_input_ids.view(-1)
        
        valid_mask = flat_labels != -100
        
        # === 4. 区域划分 ===
        is_feature_token = torch.zeros_like(flat_input_ids, dtype=torch.bool)
        for rid in self.feature_token_ids:
            is_feature_token |= (flat_input_ids == rid)
        if self.extract_token_id is not None:
             is_feature_token |= (flat_input_ids == self.extract_token_id)

        # 区域定义:
        # Latent Area: 需要做 MSE 的 Token 
        # 注意：如果 Dataset 没有插入 Latent Token（针对无图数据），这里 naturally 为 False
        is_latent_area = is_feature_token & valid_mask
        
        # Text Area: 所有非 Feature 的有效 Token
        is_text_area = (~is_feature_token) & valid_mask
        
        # === 5. 计算 Loss ===
        loss_fct = nn.CrossEntropyLoss(reduction='none') 
        ce_loss_all = loss_fct(flat_logits_v, flat_labels)
        
        # A. Latent CE Loss
        # 只在 is_latent_area 计算，如果无图数据没有 latent tokens，分母为 0，loss 为 0
        latent_ce_loss = torch.tensor(0.0, device=logits_v.device)
        if is_latent_area.sum() > 0:
            latent_ce_loss = (ce_loss_all * is_latent_area.float()).sum() / is_latent_area.sum()
        
        # B. MSE Loss
        last_hidden_state = outputs_v.hidden_states[-1] 
        total_mse_loss = torch.tensor(0.0, device=logits_v.device)
        mse_cnt = 0
        
        if alignment_features:
            for stage in self.config.stages:
                target_feat = alignment_features.get(stage.feature_key)
                if target_feat is None: continue
                
                # target_feat shape: [Batch, Dim]
                # has_visual_features shape: [Batch]
                
                stage_mask = (input_ids == stage.token_id)
                # 如果整个 batch 都没有这个 token (例如全都是无图数据)，直接跳过
                if stage_mask.sum() == 0: continue

                for b in range(input_ids.size(0)):
                    # [关键修改] 如果该样本标记为无视觉特征，跳过 MSE 计算
                    if has_visual_features is not None and not has_visual_features[b]:
                        continue
                        
                    b_mask = stage_mask[b]
                    if b_mask.sum() > 0:
                        b_hidden = last_hidden_state[b][b_mask]
                        # Mean Pool 对齐
                        b_proj = self.projectors[stage.name](b_hidden.mean(dim=0, keepdim=True))
                        target = target_feat[b].unsqueeze(0).to(dtype=b_proj.dtype, device=b_proj.device)
                        total_mse_loss += F.mse_loss(b_proj, target)
                        mse_cnt += 1
        
        if mse_cnt > 0:
            total_mse_loss /= mse_cnt

        # C. SFT Loss (普通文本 + 结构标签)
        ans_ce_loss = torch.tensor(0.0, device=logits_v.device)
        if is_text_area.sum() > 0:
            ans_ce_loss = (ce_loss_all * is_text_area.float()).sum() / is_text_area.sum()

        # D. VBC Loss
        loss_vbc = torch.tensor(0.0, device=logits_v.device)
        
        if do_vbc and logits_blind is not None:
            shift_logits_blind = logits_blind[..., :-1, :].contiguous()
            flat_logits_blind = shift_logits_blind.view(-1, shift_logits_blind.size(-1))
            
            # 确定 VBC 计算范围
            if self.config.vbc_target_scope == "answer" and shift_answer_mask is not None:
                vbc_target_mask = shift_answer_mask & valid_mask
            else:
                vbc_target_mask = is_text_area
            
            # [关键修改] 排除无图样本的 token
            if has_visual_features is not None:
                # has_visual_features: [B] -> 扩展到 token 级别
                # batch_indices: [B * SeqLen]
                batch_indices = torch.arange(input_ids.size(0), device=input_ids.device).unsqueeze(1).expand(-1, shift_input_ids.size(1)).reshape(-1)
                token_has_visual = has_visual_features[batch_indices]
                vbc_target_mask = vbc_target_mask & token_has_visual

            if vbc_target_mask.sum() > 0:
                probs_blind = F.softmax(flat_logits_blind, dim=-1)
                p_blind_gt = probs_blind.gather(1, flat_labels.unsqueeze(1)).squeeze(1)
                
                # 动态 Margin 加权
                weights = F.relu(self.config.vbc_margin - p_blind_gt)
                
                logits_diff = flat_logits_v - flat_logits_blind
                loss_vbc_raw = loss_fct(logits_diff, flat_labels)
                
                loss_vbc = (loss_vbc_raw * weights * vbc_target_mask.float()).sum() / (vbc_target_mask.sum() + 1e-8)

        # === Total ===
        total_loss = (
            latent_ce_loss +  
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