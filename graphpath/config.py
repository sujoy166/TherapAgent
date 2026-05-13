"""Hyperparameters and paths for the GraphPath model (Ma & Wang 2024)."""
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

    # ── Phenotype heads (decoded from stage = 4*TMT + 2*RT + 1*OS) ───────
    head_names: tuple = ("TMT", "RT", "OS")

    # ── Reactome adjacency ───────────────────────────────────────────────
    species_prefix: str = "R-HSA-"
    pathways_url: str = "https://reactome.org/download/current/ReactomePathways.txt"
    relations_url: str = "https://reactome.org/download/current/ReactomePathwaysRelation.txt"
    include_siblings: bool = True   # adjacency = parent/child ∪ siblings (KEGG-style)

    # ── Model (paper §2.4–2.5) ───────────────────────────────────────────
    embed_dim: int = 64             # F' — node embedding width per head
    n_heads: int = 3                # K = 3 in paper
    readout_dim: int = 1            # per-pathway scalar readout (paper §2.5)
    dropout: float = 0.4            # paper §2.6

    # ── Training (paper §2.6) ────────────────────────────────────────────
    val_frac: float = 0.10          # paper uses 80/10/10
    test_frac: float = 0.10
    seed: int = 42
    batch_size: int = 32
    lr: float = 0.05                # paper: SGD lr = 0.05
    momentum: float = 0.9
    weight_decay: float = 0.05
    max_epochs: int = 200
    patience: int = 25
    plateau_patience: int = 8
    plateau_factor: float = 0.5

    def __post_init__(self):
        self.scores_csv = self.project_root / "pathway_scores.csv"
        self.mapping_csv = self.project_root / "pathway_phenotype_mapping.csv"
        self.cache_dir = self.project_root / "graphpath" / "cache"
        self.artifacts_dir = self.project_root / "graphpath" / "artifacts"
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.artifacts_dir.mkdir(parents=True, exist_ok=True)
