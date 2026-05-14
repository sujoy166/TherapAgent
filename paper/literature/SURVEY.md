# Literature survey for the TherapAgent / ASI 2026 manuscript

Searched 2026-05-12 via WebSearch against bioRxiv, medRxiv, PubMed, Nature, arXiv. The MCP-style bioRxiv server is not directly available in this environment, so queries were issued through the harness's web tools.

## Pathway-informed deep learning (foundations)

- **P-NET (Elmarakeby et al. 2021)** — *Nature* 598:348-352. The progenitor pathway-aware multi-layered hierarchical network. 3,007 Reactome pathways; sparse mask between gene and pathway layers; layered parent-child for higher levels; predicts metastatic vs. primary prostate cancer; SHAP-based interpretation. Inspired BINN, GraphPath, PATH.
- **BINN (Hartman et al. 2023)** — *Nat. Commun.* 14:5359. Generalises P-NET to *proteomics* + sepsis/COVID-19 subphenotyping; introduces per-layer auxiliary heads (Eq. 1) and the log-subgraph-size SHAP adjustment.
- **GraphPath (Ma & Wang 2024)** — *Bioinformatics* 40:btae165. Replaces hierarchy with a *pathway-pathway interaction graph* (KEGG-derived) processed by a multi-head GAT (K=3, ELU). 511 pathways, prostate cancer CRPC vs. primary.
- **PATH (Howlader et al. 2026)** — arXiv:2604.16685. FiLM-modulated gene embeddings → attention-pooled pathway tokens → edge-aware **graph transformer** with Laplacian positional encoding, soft structural mask, per-head edge-conditioned attention bias; pan-cancer metastasis prediction.
- **Pathformer (Liu et al. 2024)** — *Bioinformatics* 40:btae316. Cross-modal multi-omics transformer with criss-cross attention over pathway tokens; +6.3-14.7% F1 over 18 baselines in cancer survival.
- **Wysocka et al. 2023** — *BMC Bioinformatics* 24:198. Systematic review of biologically-informed DL models for cancer encoding/interpretation. Identifies four design axes: gene-pathway membership, parent-child hierarchy, pathway-pathway edges, and attention/regularisation strategies.

## Therapy-response / immunotherapy prediction

- **IRnet (Jiang et al. 2024)** — *J. Adv. Res.* (S2090123224003205, PMC12147637). Pathway knowledge-informed GNN for immune-checkpoint-inhibitor response across melanoma, gastric, bladder cohorts. Three interpretability levels: pathway, pathway-interaction, gene.
- **PathHDNN (Genome Medicine 2025)** — pathway hierarchical-informed DNN for immunotherapy response + mechanism inference, validated on ICI cohorts.
- **PathNetDRP (PMC12051301, 2025)** — biomarker discovery framework using pathway + PPI networks for ICI-response prediction; demonstrates that pathway + PPI features outperform either alone.
- **PDDRNet-MH (bioRxiv 2025.06.09.658757)** — multiplex heterogeneous network for patient-derived drug response in breast cancer; integrates genomic + transcriptomic + epigenomic + drug-structure + side-effect features.
- **Generalizable AI for immunotherapy outcomes (medRxiv 2025.05.01.25326820)** — pan-cancer cross-treatment ICI response model.
- **Spatial ABM for HER2-heterogeneous BC (bioRxiv 2026.03.14.711774)** — agent-based + ML for combination-therapy response in HER2-heterogeneous breast cancer.

## ssGSEA in TCGA-BRCA

- **Barbie et al. 2009** — *Nature* 462:108. Original ssGSEA algorithm we re-implemented in `gene_to_pathway.py`.
- ssGSEA-based BRCA prognostic and immune-infiltration models (Front. Genet. 2022, Discov. Oncol. 2025, J. Transl. Med. 2019, BMC Cancer 2022) provide downstream-task baselines.
- Post-radiation BRCA DNA-methylation regulation analysis (*Sci. Rep.* 2025) — directly relevant for the RT head.

## Core dependencies

- **Reactome (Gillespie et al. 2022)** — *Nucleic Acids Res.* 50:D687. Pathway database used by all three models.
- **GSEA (Subramanian et al. 2005)** — *PNAS* 102:15545. Original GSEA framework.
- **GAT (Veličković et al. 2018)** — *ICLR*. Multi-head graph attention.
- **Graph Transformer (Dwivedi & Bresson 2021)** — Generalises self-attention to graphs; PATH builds on this.
- **FiLM (Perez et al. 2018)** — *AAAI*. Feature-wise linear modulation; PATH Stage 1.
- **SHAP (Lundberg & Lee 2017)** — *NeurIPS*. Feature attribution for BINN/PATH/GraphPath interpretability.
- **TCGA-BRCA** — *Nature* 490:61 (2012). Source cohort.

## Visualisation

- **Okabe & Ito 2008** — colour-vision-deficiency palette; Nature Methods recommended.
- **Viridis (van der Walt & Smith 2015)** — perceptually uniform sequential colormaps for matplotlib.

## Workshop framing (ASI 2026 / ACM-BCB Track II)

- Submission: 8-10 pages, ACM Master Article Template (acmart, sigconf).
- Themes that fit our paper:
  - Cancer Immunology
  - AI-driven drug discovery for immunological targets
  - Modeling and simulation of tumor-immune system interactions
  - Modeling of antigen processing, presentation, and recognition
  - Integration of multi-omics data
- Reactome's pathway taxonomy includes a large *Immune System* root with hundreds of descendants (innate / adaptive / cytokine / antigen processing / TCR / BCR signalling). Models therefore implicitly learn *immune-pathway* programmes for therapy response.
