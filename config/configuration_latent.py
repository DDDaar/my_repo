import yaml
from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any

@dataclass
class ReasoningStage:
    name: str
    token: str
    count: int
    feature_key: str
    dim: int
    text_prefix: str = "" 
    token_id: Optional[int] = None 

@dataclass
class SingleDatasetConfig:
    """单个数据集的标准化配置"""
    name: str
    split: str
    count: int          # -1 代表全量
    image_folder: str
    feature_dir: str

@dataclass
class LatentConfig:
    # === 1. 模型参数 ===
    base_model: str
    hidden_size: int
    
    # === 2. 数据参数 ===
    train_datasets: List[SingleDatasetConfig]
    seed: int
    
    # [新增] 是否要求必须有预提取的视觉特征
    # True: 过滤掉没有特征文件的样本 (旧模式)
    # False: 保留所有样本。无特征样本只计算 SFT Loss (新模式)
    require_vision_features: bool

    # === 3. Token 与 结构参数 ===
    extract_token: str
    extract_count: int
    extract_text_prefix: str 
    
    # 结构性 Token
    think_start: str
    think_end: str
    answer_start: str
    answer_end: str
    anchor_start: str
    anchor_end: str

    stages: List[ReasoningStage]
    
    # === 4. 训练参数 ===
    epochs: int          
    debug_print_steps: int
    alpha_sft: float    # 标准 CE Loss 权重
    beta_mse: float     # 特征对齐 MSE Loss 权重
    lambda_vbc: float   # 视觉瓶颈对比 Loss 权重
    vbc_margin: float   # 动态权重阈值
    
    batch_size: int
    gradient_checkpointing: bool
    
    # === 5. 运行时字段 (由 train.py 填充) ===
    extract_token_id: Optional[int] = None
    vbc_target_scope: str = "answer" 
    
    # 用于 Dataset 定位 Mask 的关键 ID
    answer_start_id: Optional[int] = None
    answer_end_id: Optional[int] = None

    use_anchor: bool = False

    @classmethod
    def load(cls, path: str):
        with open(path, 'r') as f:
            cfg = yaml.safe_load(f)
        
        # 解析各个推理阶段
        stages = []
        for s in cfg['reasoning_stages']:
            stages.append(ReasoningStage(
                name=s['name'],
                token=s['token'],
                count=s['count'],
                feature_key=s['feature_key'],
                dim=s['dim'],
                text_prefix=s.get('text_prefix', "") 
            ))
        
        # 解析数据配置
        data_cfg = cfg.get('data', {})
        mode = data_cfg.get('train_mode', 'single')
        
        unified_datasets = []
        
        if mode == 'mix' and 'mix_datasets' in data_cfg:
            for item in data_cfg['mix_datasets']:
                unified_datasets.append(SingleDatasetConfig(
                    name=item['name'],
                    split=item.get('split', 'train'),
                    count=item.get('count', -1),
                    image_folder=item.get('image_folder', ''),
                    feature_dir=item.get('feature_dir', '')
                ))
        else:
            # 单数据集兼容模式
            d_name = data_cfg.get('dataset_name', cfg.get('dataset_name'))
            d_split = data_cfg.get('dataset_split', cfg.get('dataset_split', 'train'))
            d_count = data_cfg.get('count', cfg.get('count', -1))
            d_img_root = data_cfg.get('image_folder', '') 
            d_feat_root = data_cfg.get('feature_dir', '')
            
            if d_name:
                unified_datasets.append(SingleDatasetConfig(
                    name=d_name,
                    split=d_split,
                    count=d_count,
                    image_folder=d_img_root,
                    feature_dir=d_feat_root
                ))

        tokens_cfg = cfg.get('tokens', {})

        return cls(
            base_model=cfg['model']['base_model'],
            hidden_size=cfg['model']['hidden_size'],
            train_datasets=unified_datasets,
            seed=data_cfg.get('seed', 42),
            require_vision_features=data_cfg.get('require_vision_features', True),
            
            # Token配置
            extract_token=tokens_cfg.get('extract_token', "<|vision_extract_pad|>"),
            extract_count=tokens_cfg.get('extract_count', 0),
            extract_text_prefix=tokens_cfg.get('extract_text_prefix', ""),
            
            # 读取新的标签
            think_start=tokens_cfg.get('think_start', "<think>"),
            think_end=tokens_cfg.get('think_end', "</think>"),
            answer_start=tokens_cfg.get('answer_start', "<answer>"),
            answer_end=tokens_cfg.get('answer_end', "</answer>"),
            anchor_start=tokens_cfg.get('anchor_start', "<|anchor_start|>"),
            anchor_end=tokens_cfg.get('anchor_end', "<|anchor_end|>"),

            use_anchor=tokens_cfg.get('use_anchor', False),

            stages=stages,
            
            # 训练超参
            epochs=cfg['training']['epochs'], 
            debug_print_steps=cfg['training']['debug_print_steps'], 
            alpha_sft=cfg['training']['alpha_sft'],
            beta_mse=cfg['training']['beta_mse'],
            lambda_vbc=cfg['training'].get('lambda_vbc', 0.5),
            vbc_margin=cfg['training'].get('vbc_margin', 0.8),
            vbc_target_scope=cfg['training'].get('vbc_target_scope', "answer"),
            
            batch_size=cfg['training']['batch_size'],
            gradient_checkpointing=cfg['training'].get('gradient_checkpointing', False)
        )
    
    def get_all_special_tokens(self) -> List[str]:
        """强制返回一个固定顺序的 Token 列表，确保 ID 永不偏移"""
        tokens = [
            self.think_start,
            self.think_end,
            self.answer_start,
            self.answer_end,
            self.extract_token
        ]
        # 加上推理阶段的 token
        for stage in self.stages:
            if stage.token not in tokens:
                tokens.append(stage.token)
        if self.use_anchor:
            tokens.extend([self.anchor_start, self.anchor_end])
        return tokens