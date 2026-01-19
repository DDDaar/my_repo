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
            attn_implementation="sdpa" # 推荐使用 SDPA 或 FlashAttn
        )

        self.projectors = nn.ModuleDict({
            stage.name: nn.Linear(config_obj.hidden_size, stage.dim)
            for stage in config_obj.stages
        })
        
        # 缓存特殊的 Token IDs 以便在 Forward 中快速查找
        self.pad_token_ids = set()
        if config_obj.extract_token_id:
            self.pad_token_ids.add(config_obj.extract_token_id)
        for s in config_obj.stages:
            self.pad_token_ids.add(s.token_id)

    def _create_custom_mask(self, input_ids, attention_mask):
        """
        构建自定义 4D Mask [Batch, 1, Seq, Seq]
        规则：
        1. 基本是 Causal Mask (下三角)。
        2. 图像 Token 对 Text (Question/Answer) 不可见。
        3. 图像 Token 对 Pad (Extract/Reasoning) 可见。
        """
        B, L = input_ids.shape
        device = input_ids.device
        
        # 1. 基础 Causal Mask
        # min_dtype: float16/bfloat16 的极小值
        min_dtype = torch.finfo(self.base_model.dtype).min
        
        # 创建下三角掩码 (1 for allow, 0 for block)
        # causal_mask[i, j] = 1 if i >= j
        causal_mask = torch.tril(torch.ones((L, L), device=device, dtype=torch.bool))
        
        # 扩展 padding mask (input 里的 attention_mask 是 padding mask)
        # attention_mask: [B, L], 1 is valid, 0 is padding
        # [B, 1, 1, L]
        padding_mask = attention_mask.view(B, 1, 1, L).bool()
        
        # 2. 识别 Image Token 区域
        # Qwen2.5-VL 使用 <|vision_start|> (151652) 和 <|vision_end|> (151653) 包裹图像
        # 或者我们可以根据 token ID 范围判断（较复杂），或者假设 input_ids 中连续的大量特殊 token 是图像
        # 这里我们使用一种通用的假设：Qwen 的 vision tokens 是特殊的 placeholders
        # 为了精确控制，我们需要 processor 的 tokenizer 来获取 vision_start/end id
        # 假设 config 里没存，我们硬编码 Qwen2.5-VL 的默认值 (需确认)
        VISION_START_ID = 151652
        VISION_END_ID = 151653
        
        # 3. 识别 Pad Token 区域 (Extract + Reasoning)
        # 创建一个 Boolean Map: [B, L] True if token is a Pad Token
        is_pad_token = torch.zeros_like(input_ids, dtype=torch.bool)
        for tid in self.pad_token_ids:
            is_pad_token |= (input_ids == tid)

        # 4. 构建最终 Mask
        # 目标: [B, 1, L, L] -> Float tensor
        # 初始化为 full mask (all blocked)
        final_mask = torch.full((B, 1, L, L), min_dtype, device=device, dtype=self.base_model.dtype)
        
        for b in range(B):
            # 获取当前样本的 Image 范围
            # 找到 vision_start 和 vision_end 的位置
            v_starts = (input_ids[b] == VISION_START_ID).nonzero(as_tuple=True)[0]
            v_ends = (input_ids[b] == VISION_END_ID).nonzero(as_tuple=True)[0]
            
            img_indices = []
            if len(v_starts) > 0 and len(v_ends) > 0:
                # 假设只有一张图或多张图
                for start, end in zip(v_starts, v_ends):
                    # 包含 start 和 end 本身以及中间的所有 tokens
                    img_indices.extend(range(start.item(), end.item() + 1))
            
            img_indices_tensor = torch.tensor(img_indices, device=device)
            
            # 逻辑：
            # 基础是 Causal: mask[i, j] = 0 if i >= j else -inf
            # 加上 Padding: mask[i, j] = -inf if padding_mask[b, j] == 0
            
            # 先填充标准的 Causal Mask
            current_causal = torch.zeros((L, L), device=device, dtype=self.base_model.dtype)
            current_causal.masked_fill_(~causal_mask, min_dtype)
            
            # 应用 Padding (Key Padding)
            # padding_mask[b] is [1, 1, L] -> [L]
            pad_row = attention_mask[b] == 0 # 1 is valid, 0 is pad
            current_causal[:, pad_row] = min_dtype
            
            # === 核心约束 ===
            if len(img_indices) > 0:
                # 谁是 Pad Token (可以看图)
                # 谁是 Text Token (不能看图) -> 即 (NOT Pad Token) AND (NOT Image Token)
                # 注意：Image Token 自己当然可以看自己，所以我们只限制 Text Token
                
                # 当前样本的 Pad Mask
                b_is_pad = is_pad_token[b] # [L]
                
                # 哪些行（Query）是"文本" (既不是Pad也不是Image本身)
                # 注意：Question 和 Answer 都是文本
                # 为了简化，我们令 mask[Row, Col] = -inf
                # 其中 Row 是 Text, Col 是 Image
                
                # 构建 Row Mask (Text Tokens)
                is_img_token = torch.zeros(L, device=device, dtype=torch.bool)
                is_img_token[img_indices_tensor] = True
                
                # Text = Not Pad AND Not Image
                is_text_row = (~b_is_pad) & (~is_img_token)
                
                # 执行屏蔽：Text Rows 不能 attend 到 Image Cols
                # 利用广播机制设置
                # current_causal[is_text_row][:, is_img_token] = min_dtype # 这种索引方式在 pytorch 某些版本不支持二维同时索引
                
                # 使用 fill 
                # 创建一个 [L, L] 的 block mask
                # block_map[i, j] = True 表示 i 是 Text 且 j 是 Image
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
        # 1. 构建自定义 Mask
        # 此时的 attention_mask 将是一个 4D tensor，直接传给 model 会覆盖其内部的 causal mask 生成逻辑
        custom_mask = self._create_custom_mask(input_ids, attention_mask)
        
        # 2. Base Model Forward
        outputs = self.base_model(
            input_ids=input_ids,
            attention_mask=custom_mask, # 传入 4D mask
            pixel_values=pixel_values,
            image_grid_thw=image_grid_thw,
            labels=labels, 
            output_hidden_states=True,
            return_dict=True,
        )

        sft_loss = outputs.loss
        last_hidden_state = outputs.hidden_states[-1]

        # 3. 计算 MSE Loss (仅针对 Reasoning Stages)
        total_mse_loss = 0.0
        mse_count = 0

        # 注意：Extract Token (vision_extract_pad) 不计算 MSE
        # 它只负责汇聚信息，通过 Reasoning Stages 的梯度回传进行学习
        
        for stage in self.config.stages:
            target_feat = alignment_features.get(stage.feature_key)
            if target_feat is None: continue
                
            stage_mask = (input_ids == stage.token_id)
            if stage_mask.sum() == 0: continue

            # 提取特征并对齐
            # 策略：对每个样本的该阶段所有 tokens 取平均，然后与 global 特征对齐
            # 或者如果 target_feat 是序列 (如 depth map patches)，需要更复杂的对齐
            # 这里假设 target_feat 是 global vector [Batch, Dim]
            
            # 按 Batch 计算以保证正确匹配
            batch_loss = 0.0
            valid_b = 0
            
            for b in range(input_ids.size(0)):
                b_mask = stage_mask[b]
                if b_mask.sum() > 0:
                    # [N_tokens, Hidden]
                    b_hidden = last_hidden_state[b][b_mask]
                    # Average -> [1, Hidden]
                    b_pooled = b_hidden.mean(dim=0, keepdim=True)
                    # Project -> [1, Target_Dim]
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