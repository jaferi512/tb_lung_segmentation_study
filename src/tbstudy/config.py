"""Experiment configuration.

Every value marked [paper] is taken from Haider et al. (2026), Pak J Med Cardiol Rev 5(1):7719-7746,
Methods section and Table 5. Values marked [assumption] are not specified in the paper and were
chosen by this reimplementation.
"""
from dataclasses import dataclass, field, asdict
from pathlib import Path


@dataclass
class SegConfig:
    input_size: int = 128            # [paper] U-Net trained at 128x128 (area downsampling)
    eval_size: int = 256             # [paper] probability maps upsampled to 256 for evaluation
    base_channels: int = 16          # [paper] 16 -> 256 filters, four levels + bottleneck
    epochs: int = 50                 # [paper]
    batch_size: int = 8              # [paper]
    lr: float = 1e-3                 # [paper] Adam + cosine annealing
    rotation_deg: float = 10.0       # [paper]
    scale: float = 0.10              # [paper]
    shift: float = 0.05              # [paper] fraction of image size
    brightness: float = 0.10         # [assumption] paper says "brightness and contrast changes"
    contrast: float = 0.10           # [assumption]
    threshold: float = 0.5           # [paper]
    keep_components: int = 2         # [paper] two largest connected components


@dataclass
class ClsConfig:
    input_size: int = 224            # [paper]
    epochs: int = 15                 # [paper]
    batch_size: int = 16             # [paper]
    lr: float = 1e-4                 # [paper] AdamW
    weight_decay: float = 1e-4       # [paper]
    rotation_deg: float = 7.0        # [paper]
    scale: float = 0.08              # [paper]
    shift: float = 0.04              # [paper]
    brightness: float = 0.05         # [assumption] paper says "small brightness and contrast changes"
    contrast: float = 0.05           # [assumption]
    roi_dilation_px: int = 6         # [paper] at 256x256
    threshold: float = 0.5           # [paper]
    pretrained: bool = True          # [paper] ImageNet initialisation


@dataclass
class EvalConfig:
    n_bootstrap: int = 2000          # [paper]
    bootstrap_seed: int = 2026       # [assumption]
    target_sensitivity: float = 0.90  # [paper] WHO triage minimum
    target_specificity_who: float = 0.70  # [paper] WHO triage minimum
    high_specificity: float = 0.95   # [paper]
    prevalences: tuple = (0.01, 0.05, 0.10)  # [paper]
    # Keyword subgroups on the clinical readings [paper: "grouped by keywords"; exact list is an assumption]
    pleural_keywords: tuple = ("pleur", "effusion")
    inactive_keywords: tuple = ("inactive", "scar", "calcif", "stable")


@dataclass
class Config:
    data_dir: Path = Path("data/MontgomerySet")
    out_dir: Path = Path("outputs")
    preprocess_size: int = 256       # [paper]
    n_folds: int = 5                 # [paper]
    seed: int = 42                   # [paper] fold split seed; also used for network initialisation
    num_workers: int = 0
    device: str = "auto"
    seg: SegConfig = field(default_factory=SegConfig)
    cls: ClsConfig = field(default_factory=ClsConfig)
    eval: EvalConfig = field(default_factory=EvalConfig)

    def to_dict(self):
        d = asdict(self)
        d["data_dir"] = str(self.data_dir)
        d["out_dir"] = str(self.out_dir)
        return d

    # Convenience paths
    @property
    def cache_file(self) -> Path:
        return self.out_dir / "cache" / f"montgomery_{self.preprocess_size}.npz"

    @property
    def meta_file(self) -> Path:
        return self.out_dir / "cache" / "metadata.csv"

    @property
    def seg_dir(self) -> Path:
        return self.out_dir / "segmentation"

    @property
    def cls_dir(self) -> Path:
        return self.out_dir / "classification"

    @property
    def results_dir(self) -> Path:
        return self.out_dir / "results"


def smoke_config(data_dir: Path, out_dir: Path) -> Config:
    """Tiny settings used by `python -m tbstudy smoke` to check the pipeline end to end in minutes."""
    cfg = Config(data_dir=data_dir, out_dir=out_dir)
    cfg.seg.epochs = 2
    cfg.cls.epochs = 2
    cfg.cls.pretrained = False
    cfg.eval.n_bootstrap = 50
    return cfg
