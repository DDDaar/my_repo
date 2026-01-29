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
    # 阶段前的文本引导，例如 " Then analyze semantic: "
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

    # === 3. Token 与 前缀参数 ===
    extract_token: str
    extract_count: int
    # [新增] 提取阶段的自然语言引导前缀
    extract_text_prefix: str 

    stages: List[ReasoningStage]
    
    # === 4. 训练参数 ===
    epochs: int          
    alpha_sft: float    # 标准 CE Loss 权重
    beta_mse: float     # 特征对齐 MSE Loss 权重
    
    # [新增] VBC 相关参数
    lambda_vbc: float   # 视觉瓶颈对比 Loss 权重
    vbc_margin: float   # 动态权重阈值
    
    batch_size: int
    gradient_checkpointing: bool
    
    # === 5. 运行时字段 ===
    extract_token_id: Optional[int] = None
    
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
            else:
                # 兜底防止报错，实际使用需配置正确
                pass

        return cls(
            base_model=cfg['model']['base_model'],
            hidden_size=cfg['model']['hidden_size'],
            train_datasets=unified_datasets,
            seed=data_cfg.get('seed', 42),
            
            # Token配置
            extract_token=cfg['tokens']['extract_token'],
            extract_count=cfg['tokens']['extract_count'],
            extract_text_prefix=cfg['tokens'].get('extract_text_prefix', ""),
            
            stages=stages,
            
            # 训练超参
            epochs=cfg['training']['epochs'], 
            alpha_sft=cfg['training']['alpha_sft'],
            beta_mse=cfg['training']['beta_mse'],
            lambda_vbc=cfg['training'].get('lambda_vbc', 0.5),
            vbc_margin=cfg['training'].get('vbc_margin', 0.8),
            
            batch_size=cfg['training']['batch_size'],
            gradient_checkpointing=cfg['training'].get('gradient_checkpointing', False)
        )