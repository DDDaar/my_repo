import yaml
from dataclasses import dataclass
from typing import List, Optional

@dataclass
class ReasoningStage:
    name: str
    token: str
    count: int
    feature_key: str
    dim: int
    token_id: Optional[int] = None 

@dataclass
class LatentConfig:
    # 1. 模型参数
    base_model: str
    hidden_size: int
    
    # 2. 数据参数 (新增)
    dataset_name: str
    dataset_split: str

    # 3. Token 参数
    extract_token: str
    extract_count: int
    
    stages: List[ReasoningStage]
    
    # 4. 训练参数
    epochs: int          
    alpha_sft: float
    beta_mse: float
    batch_size: int
    
    # 5. 选填字段
    gradient_checkpointing: bool = False
    extract_token_id: Optional[int] = None
    
    @classmethod
    def load(cls, path: str):
        with open(path, 'r') as f:
            cfg = yaml.safe_load(f)
        
        stages = [ReasoningStage(**s) for s in cfg['reasoning_stages']]
        
        # 兼容旧配置文件的处理
        data_cfg = cfg.get('data', {})
        # 默认回退到 ScienceQA 以防报错，但建议更新 yaml
        d_name = data_cfg.get('dataset_name', "derek-thomas/ScienceQA")
        d_split = data_cfg.get('dataset_split', "train")

        return cls(
            base_model=cfg['model']['base_model'],
            hidden_size=cfg['model']['hidden_size'],
            
            dataset_name=d_name,
            dataset_split=d_split,

            extract_token=cfg['tokens']['extract_token'],
            extract_count=cfg['tokens']['extract_count'],
            stages=stages,
            
            epochs=cfg['training']['epochs'], 
            alpha_sft=cfg['training']['alpha_sft'],
            beta_mse=cfg['training']['beta_mse'],
            batch_size=cfg['training']['batch_size'],
            gradient_checkpointing=cfg['training'].get('gradient_checkpointing', False)
        )