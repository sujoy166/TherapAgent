"""Hyperparameters and paths for the PATH model (Howlader et al. 2026)."""
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class Config:
    # ── Paths ────────────────────────────────────────────────────────────
    project_root: Path = Path(__file__).resolve().parent.parent
    scores_csv: Path = field(init=False)
    mapping_csv: Path = field(init=False)
    cache_dir: Path = field(init=False)
    artifacts_dir: Path = field(init=False)

    head_names: tuple = ("TMT", "RT", "OS")

    # ── Reactome adjacency (Jaccard from GMT gene memberships) ───────────
    gmt_url: str = "https://reactome.org/download/current/ReactomePathways.gmt.zip"
    pathways_url: str = "https://reactome.org/download/current/ReactomePathways.txt"
    species_prefix: str = "R-HSA-"
    min_pathway_size: int = 15      # paper §4.1
    jaccard_threshold: float = 0.0  # keep all positive Jaccard edges

    # ── Model (paper §4.2) ───────────────────────────────────────────────
    embed_dim: int = 64             # d
    n_heads: int = 4                # H
    n_layers: int = 2               # L (graph transformer blocks)
    ffn_expansion: int = 4
    laplacian_k: int = 16           # k positional eigenvectors (truncated if N<k)
    soft_mask_penalty: float = -10.0
    dropout: float = 0.2

    # ── Training (paper §4.3) ────────────────────────────────────────────
    val_frac: float = 0.10          # paper: 72/8/20; we use 80/10/10 for our cohort
    test_frac: float = 0.10
    seed: int = 42
    batch_size: int = 16            # paper §4.3
    lr: float = 1e-4                # AdamW, paper §4.3
    weight_decay: float = 5e-4
    grad_clip: float = 2.0
    max_epochs: int = 200
    min_epochs: int = 25            # paper §4.3 sets a 50-epoch min; we use 25 due to smaller cohort
    patience: int = 25
    plateau_patience: int = 10
    plateau_factor: float = 0.5

    def __post_init__(self):
        self.scores_csv = self.project_root / "pathway_scores.csv"
        self.mapping_csv = self.project_root / "pathway_phenotype_mapping.csv"
        self.cache_dir = self.project_root / "path" / "cache"
        self.artifacts_dir = self.project_root / "path" / "artifacts"
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.artifacts_dir.mkdir(parents=True, exist_ok=True)
