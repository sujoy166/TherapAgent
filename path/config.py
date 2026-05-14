"""Hyperparameters and paths for the PATH model (Howlader et al. 2026)."""
from dataclasses import dataclass, field
from pathlib import Path

from binn.config import COHORT_FILES


@dataclass
class Config:
    cohort: str = "breast"

    project_root: Path = Path(__file__).resolve().parent.parent
    scores_csv: Path = field(init=False)
    mapping_csv: Path = field(init=False)
    cache_dir: Path = field(init=False)
    artifacts_dir: Path = field(init=False)

    head_names: tuple = ("TMT", "RT", "OS")

    gmt_url: str = "https://reactome.org/download/current/ReactomePathways.gmt.zip"
    pathways_url: str = "https://reactome.org/download/current/ReactomePathways.txt"
    species_prefix: str = "R-HSA-"
    min_pathway_size: int = 15
    jaccard_threshold: float = 0.0

    embed_dim: int = 64
    n_heads: int = 4
    n_layers: int = 2
    ffn_expansion: int = 4
    laplacian_k: int = 16
    soft_mask_penalty: float = -10.0
    dropout: float = 0.2

    val_frac: float = 0.10
    test_frac: float = 0.10
    seed: int = 42
    batch_size: int = 16
    lr: float = 1e-4
    weight_decay: float = 5e-4
    grad_clip: float = 2.0
    max_epochs: int = 200
    min_epochs: int = 25
    patience: int = 25
    plateau_patience: int = 10
    plateau_factor: float = 0.5

    def __post_init__(self):
        if self.cohort not in COHORT_FILES:
            raise ValueError(
                f"unknown cohort {self.cohort!r}. "
                f"Available: {sorted(COHORT_FILES)}"
            )
        intermediate, final = COHORT_FILES[self.cohort]
        self.scores_csv = (
            self.project_root / "Intermediate Dataset" / f"{intermediate}.csv"
        )
        self.mapping_csv = (
            self.project_root / "Final DataSet" / f"{final}.csv"
        )
        self.cache_dir = self.project_root / "path" / "cache"
        self.artifacts_dir = (
            self.project_root / "path" / "artifacts" / self.cohort
        )
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.artifacts_dir.mkdir(parents=True, exist_ok=True)
