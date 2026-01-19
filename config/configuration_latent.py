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
    # 有默认值的放在最后
    token_id: Optional[int] = None 

@dataclass
class LatentConfig:
    # 1. 必填字段 (无默认值)
    base_model: str
    hidden_size: int
    
    extract_token: str
    extract_count: int
    
    stages: List[ReasoningStage]
    alpha_sft: float
    beta_mse: float
    batch_size: int
    
    # 2. 选填字段 (有默认值) - 必须放在必填字段之后
    extract_token_id: Optional[int] = None
    
    @classmethod
    def load(cls, path: str):
        with open(path, 'r') as f:
            cfg = yaml.safe_load(f)
        
        stages = [ReasoningStage(**s) for s in cfg['reasoning_stages']]
        return cls(
            base_model=cfg['model']['base_model'],
            hidden_size=cfg['model']['hidden_size'],
            extract_token=cfg['tokens']['extract_token'],
            extract_count=cfg['tokens']['extract_count'],
            stages=stages,
            alpha_sft=cfg['training']['alpha_sft'],
            beta_mse=cfg['training']['beta_mse'],
            batch_size=cfg['training']['batch_size']
        )