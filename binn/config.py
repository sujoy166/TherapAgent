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

    # ── Reactome hierarchy ───────────────────────────────────────────────
    species_prefix: str = "R-HSA-"
    pathways_url: str = "https://reactome.org/download/current/ReactomePathways.txt"
    relations_url: str = "https://reactome.org/download/current/ReactomePathwaysRelation.txt"
    n_hidden_layers: int = 4

    # ── Training ─────────────────────────────────────────────────────────
    val_frac: float = 0.15
    test_frac: float = 0.15
    seed: int = 42
    batch_size: int = 32
    lr: float = 1e-3
    weight_decay: float = 1e-3
    max_epochs: int = 200
    patience: int = 20
    plateau_patience: int = 7
    plateau_factor: float = 0.5
    dropout: float = 0.2

    def __post_init__(self):
        self.scores_csv = self.project_root / "pathway_scores.csv"
        self.mapping_csv = self.project_root / "pathway_phenotype_mapping.csv"
        self.cache_dir = self.project_root / "binn" / "cache"
        self.artifacts_dir = self.project_root / "binn" / "artifacts"
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.artifacts_dir.mkdir(parents=True, exist_ok=True)
