from omegaconf import OmegaConf
import sys
from dataclasses import dataclass, field

@dataclass
class ModelParams:
    feat_dim: int = 32
    n_offsets: int = 10
    fork: int = 2
    use_feat_bank: bool = False
    source_path: str = ""
    model_path: str = ""
    images: str = "images"
    resolution: int = 1
    exp_name: str = ''
    white_background: bool = False
    random_background: bool = False
    resolution_scales: list = field(default_factory=lambda: [1.0])
    data_device: str = "cuda"
    eval: bool = False
    ds: int = 1
    ratio: int = 1
    undistorted: bool = False
    appearance_dim: int = 32
    add_opacity_dist: bool = False
    add_cov_dist: bool = False
    add_color_dist: bool = False
    add_level: bool = False
    extend: float = 1.1
    dist2level: str = 'round'
    base_layer: int = -1
    visible_threshold: float = 0.0
    update_ratio: float = 0.2
    progressive: bool = False
    dist_ratio: float = 0.999
    levels: int = -1
    init_level: int = -1
    extra_ratio: float = 0.25
    extra_up: float = 0.01

@dataclass
class PipelineParams:
    compute_cov3D_python: bool = False
    debug: bool = False

@dataclass
class OptimizationParams:
    iterations: int = 40000
    position_lr_init: float = 0.0
    position_lr_final: float = 0.0
    position_lr_delay_mult: float = 0.01
    position_lr_max_steps: int = 40000
    offset_lr_init: float = 0.01
    offset_lr_final: float = 0.0001
    offset_lr_delay_mult: float = 0.01
    offset_lr_max_steps: int = 40000
    feature_lr: float = 0.0075
    opacity_lr: float = 0.02
    scaling_lr: float = 0.007
    rotation_lr: float = 0.002
    mlp_opacity_lr_init: float = 0.002
    mlp_opacity_lr_final: float = 0.00002
    mlp_opacity_lr_delay_mult: float = 0.01
    mlp_opacity_lr_max_steps: int = 40000
    mlp_cov_lr_init: float = 0.004
    mlp_cov_lr_final: float = 0.004
    mlp_cov_lr_delay_mult: float = 0.01
    mlp_cov_lr_max_steps: int = 40000
    mlp_color_lr_init: float = 0.008
    mlp_color_lr_final: float = 0.00005
    mlp_color_lr_delay_mult: float = 0.01
    mlp_color_lr_max_steps: int = 40000
    mlp_featurebank_lr_init: float = 0.01
    mlp_featurebank_lr_final: float = 0.00001
    mlp_featurebank_lr_delay_mult: float = 0.01
    mlp_featurebank_lr_max_steps: int = 40000
    appearance_lr_init: float = 0.05
    appearance_lr_final: float = 0.0005
    appearance_lr_delay_mult: float = 0.01
    appearance_lr_max_steps: int = 40000
    percent_dense: float = 0.01
    lambda_dssim: float = 0.2
    start_stat: int = 500
    update_from: int = 1500
    coarse_iter: int = 10000
    coarse_factor: float = 1.5
    update_interval: int = 100
    update_until: float = 20000
    update_anchor: bool = True
    min_opacity: float = 0.005
    success_threshold: float = 0.8
    densify_grad_threshold: float = 0.0002

@dataclass
class MiscParams:
    ip: str = '127.0.0.1'
    port: int = 6009
    debug_from: int = -1
    detect_anomaly: bool = False
    warmup: bool = False
    use_wandb: bool = True
    test_iterations: list = field(default_factory=lambda: [-1])
    save_iterations: list = field(default_factory=lambda: [-1])
    quiet: bool = False
    checkpoint_iterations: list = field(default_factory=list)
    start_checkpoint: str = ''
    gpu: str = '-1'

def get_cfg(cfg_path=None, cli_args=None):
    # 加载默认配置
    cfg = OmegaConf.create({
        "model": OmegaConf.structured(ModelParams),
        "pipeline": OmegaConf.structured(PipelineParams),
        "optim": OmegaConf.structured(OptimizationParams),
        "misc": OmegaConf.structured(MiscParams),
    })
    # 加载yaml配置
    if cfg_path:
        yaml_cfg = OmegaConf.load(cfg_path)
        cfg = OmegaConf.merge(cfg, yaml_cfg)
    # 合并命令行参数
    if cli_args:
        cli_cfg = OmegaConf.from_dotlist(cli_args)
        cfg = OmegaConf.merge(cfg, cli_cfg)
    return cfg