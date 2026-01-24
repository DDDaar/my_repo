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
    # 1. 模型参数
    base_model: str
    hidden_size: int
    
    # 2. 数据统一接口
    # 无论单数据还是混合，最终都解析为这个列表供 Dataset 类使用
    train_datasets: List[SingleDatasetConfig]
    seed: int

    # 3. Token 参数
    extract_token: str
    extract_count: int
    stages: List[ReasoningStage]
    
    # 4. 训练参数
    epochs: int          
    alpha_sft: float
    beta_mse: float
    batch_size: int
    gradient_checkpointing: bool
    
    # 5. Attention 控制
    image_visible_to: List[str]

    # 6. 运行时字段
    extract_token_id: Optional[int] = None
    
    @classmethod
    def load(cls, path: str):
        with open(path, 'r') as f:
            cfg = yaml.safe_load(f)
        
        # 解析推理阶段
        stages = [ReasoningStage(**s) for s in cfg['reasoning_stages']]
        
        # === 核心修改：统一单/混数据逻辑 ===
        data_cfg = cfg.get('data', {})
        mode = data_cfg.get('train_mode', 'single') # 默认为 single 保持兼容
        
        unified_datasets = []
        
        if mode == 'mix' and 'mix_datasets' in data_cfg:
            # 混合模式
            for item in data_cfg['mix_datasets']:
                unified_datasets.append(SingleDatasetConfig(
                    name=item['name'],
                    split=item.get('split', 'train'),
                    count=item.get('count', -1),
                    image_folder=item.get('image_folder', ''),
                    feature_dir=item.get('feature_dir', '')
                ))
        else:
            # 单数据集模式 (回退到根目录或 data 目录下的配置)
            # 优先读 data 下的，如果没有读根下的 (兼容旧 yaml)
            d_name = data_cfg.get('dataset_name', cfg.get('dataset_name'))
            d_split = data_cfg.get('dataset_split', cfg.get('dataset_split', 'train'))
            # 这里的 image_folder 和 feature_dir 需要用户在单模式下配置好
            d_img_root = data_cfg.get('image_folder', '') 
            d_feat_root = data_cfg.get('feature_dir', '')
            
            if d_name:
                unified_datasets.append(SingleDatasetConfig(
                    name=d_name,
                    split=d_split,
                    count=-1,
                    image_folder=d_img_root,
                    feature_dir=d_feat_root
                ))
            else:
                raise ValueError("Config invalid: No dataset_name found for single mode.")

        att_cfg = cfg.get('attention_control', {})
        
        return cls(
            base_model=cfg['model']['base_model'],
            hidden_size=cfg['model']['hidden_size'],
            
            train_datasets=unified_datasets,
            seed=data_cfg.get('seed', 42),

            extract_token=cfg['tokens']['extract_token'],
            extract_count=cfg['tokens']['extract_count'],
            stages=stages,
            
            epochs=cfg['training']['epochs'], 
            alpha_sft=cfg['training']['alpha_sft'],
            beta_mse=cfg['training']['beta_mse'],
            batch_size=cfg['training']['batch_size'],
            gradient_checkpointing=cfg['training'].get('gradient_checkpointing', False),
            
            image_visible_to=att_cfg.get('image_visible_to', ["extract_token", "reasoning_tokens"])
        )