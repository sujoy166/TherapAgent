"""Hyperparameters and paths for the GraphPath model (Ma & Wang 2024)."""
import os
from dataclasses import dataclass, field
from pathlib import Path

# Cohort registry is shared with the BINN config; importing keeps the two
# in lock-step so any new dataset only needs to be added in one place.
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

    species_prefix: str = "R-HSA-"
    pathways_url: str = "https://reactome.org/download/current/ReactomePathways.txt"
    relations_url: str = "https://reactome.org/download/current/ReactomePathwaysRelation.txt"
    include_siblings: bool = True

    embed_dim: int = 64
    n_heads: int = 3
    readout_dim: int = 1
    dropout: float = 0.4

    val_frac: float = 0.10
    test_frac: float = 0.10
    seed: int = field(default_factory=lambda: int(os.environ.get("THERAP_SEED", "42")))
    batch_size: int = 32
    lr: float = 0.05
    momentum: float = 0.9
    weight_decay: float = 0.05
    max_epochs: int = 200
    patience: int = 25
    plateau_patience: int = 8
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
        self.cache_dir = self.project_root / "graphpath" / "cache"
        self.artifacts_dir = (
            self.project_root / "graphpath" / "artifacts" / self.cohort
        )
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.artifacts_dir.mkdir(parents=True, exist_ok=True)
