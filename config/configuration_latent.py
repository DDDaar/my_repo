import yaml
from dataclasses import dataclass
from typing import List, Dict

@dataclass
class ReasoningStage:
    name: str
    steps: int
    feature_key: str
    dim: int

@dataclass
class LatentConfig:
    base_model: str
    hidden_size: int
    type1_count: int
    latent_start_token: str
    type1_base_token: str
    stages: List[ReasoningStage]
    alpha_sft: float
    beta_mse: float
    batch_size: int
    vis_mask: Dict[str, bool] # 新增掩码配置
    latent_start_id: int = None

    @classmethod
    def load(cls, path: str):
        with open(path, 'r') as f:
            cfg = yaml.safe_load(f)
        stages = [ReasoningStage(**s) for s in cfg['reasoning_stages']]
        return cls(
            base_model=cfg['model']['base_model'],
            hidden_size=cfg['model']['hidden_size'],
            type1_count=cfg['tokens']['type1_count'],
            latent_start_token=cfg['tokens']['special_tokens']['latent_start'],
            type1_base_token=cfg['tokens']['special_tokens']['type1_base'],
            stages=stages,
            alpha_sft=cfg['training']['alpha_sft'],
            beta_mse=cfg['training']['beta_mse'],
            batch_size=cfg['training']['batch_size'],
            vis_mask=cfg['visibility_mask']
        )