from dataclasses import dataclass, field
from pathlib import Path


# Registry of (intermediate-csv stem, final-csv stem) per cohort.
# Both files live under the project root in the directories created by the
# data-curation step:
#   Intermediate Dataset/<intermediate>.csv  -- 1,706 pathways x N samples
#   Final DataSet/<final>.csv                -- N labeled samples x (sample, stage, 1706 pathway scores)
#
# Disabled cohorts (not in the registry):
#   "lung"    -- Intermediate Dataset/Lung_cancer.csv is byte-identical to the
#                breast intermediate; regenerate upstream before enabling.
#   "bladder" -- Final DataSet/Bladder_cancer_final.csv only has 21 samples;
#                too small to train a stratified split safely.
COHORT_FILES: dict = {
    "breast":    ("Breast_cancer",    "Breast_Cancer_final"),
    "prostate":  ("Prostate_cancer",  "Prostate_cancer_final"),
    "head_neck": ("Head_Neck_Cancer", "Head_Neck_Cancer_Final"),
    "thyroid":   ("Thyroid_Cancer",   "Thyroid_Cancer_Final"),
}


@dataclass
class Config:
    # ── Selected cohort ─────────────────────────────────────────────────
    cohort: str = "breast"

    # ── Paths (auto-resolved from cohort) ───────────────────────────────
    project_root: Path = Path(__file__).resolve().parent.parent
    scores_csv: Path = field(init=False)
    mapping_csv: Path = field(init=False)
    cache_dir: Path = field(init=False)
    artifacts_dir: Path = field(init=False)

    # ── Phenotype heads (decoded from stage = 4*TMT + 2*RT + 1*OS) ──────
    head_names: tuple = ("TMT", "RT", "OS")

    # ── Reactome hierarchy ──────────────────────────────────────────────
    species_prefix: str = "R-HSA-"
    pathways_url: str = "https://reactome.org/download/current/ReactomePathways.txt"
    relations_url: str = "https://reactome.org/download/current/ReactomePathwaysRelation.txt"
    n_hidden_layers: int = 4

    # ── Training ────────────────────────────────────────────────────────
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
        # Reactome cache is shared across cohorts; artifacts are per-cohort.
        self.cache_dir = self.project_root / "binn" / "cache"
        self.artifacts_dir = (
            self.project_root / "binn" / "artifacts" / self.cohort
        )
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.artifacts_dir.mkdir(parents=True, exist_ok=True)
