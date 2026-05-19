# IV. MODELS

## Overview

All three architectures address the same task: predicting therapy-response phenotypes (TMT, RT, OS ≥ 180d) from Reactome ssGSEA scores. They share:
- **Input:** One standardised ssGSEA score per Reactome-matched pathway (1,706 pathways)
- **Objective:** Multi-task learning via class-weighted binary cross-entropy on three binary prediction heads
- **Data pipeline:** Identical preprocessing and train/validation/test splits

Their architectural differences lie entirely in how biological structure is encoded as an inductive bias (Figure 1). We now describe each model in detail.

---

## A. BINN — Sparse Reactome Hierarchy

**Reference:** Hartman et al., 2023 (Nat. Commun.)

### Architecture Overview

BINN imposes the Reactome parent-child hierarchy as sparse masked-linear weights, with multi-layer auxiliary supervision.

### Pathway Hierarchy

Starting from 1,706 Reactome-matched input pathways, the model traverses parent-child edges upward through three hidden layers, yielding a 4-layer network:

$$1706 \to 656 \to 306 \to 163$$

Pathways that terminate before maximum depth are copied forward to preserve a node at every layer.

### Sparse Masked Weights

Between consecutive layers $\ell$ and $\ell+1$, the Reactome structure defines a binary connectivity mask $M^{(\ell)} \in \{0,1\}^{|L_{\ell+1}| \times |L_\ell|}$, where:

$$M^{(\ell)}_{ij} = 1 \iff \text{node } j \text{ in layer } \ell \text{ is a Reactome child of node } i \text{ in layer } \ell+1,$$
$$\text{or node } j \text{ is a terminal copied forward to position } i.$$

Masked weights are initialised with Kaiming uniform and zeroed outside the mask before the first forward pass. The runtime product $W^{(\ell)} \odot M^{(\ell)}$ enforces sparsity throughout training.

### Forward Pass and Hidden Layers

Each hidden block applies the masked transform followed by Tanh, Batch Normalisation, and Dropout ($p=0.2$):

$$h^{(\ell+1)} = \text{Dropout}\left(\text{BN}\left(\tanh\left(W^{(\ell)} \odot M^{(\ell)}\right) h^{(\ell)} + b^{(\ell)}\right)\right) \quad (1)$$

### Multi-Layer Supervision

An auxiliary classifier head $\text{Linear}(|L_\ell| \to 3)$ is attached after every layer, including the raw input ($\ell=0$). This produces three logits per layer (one per prediction head). The final predicted probability for head $h$ is the mean of all four sigmoid outputs:

$$\hat{p}_h(x) = \frac{1}{4} \sum_{\ell=0}^{3} \sigma\left(H^{(\ell)}(h^{(\ell)})[h]\right) \quad (2)$$

where $H^{(\ell)}$ is the classifier for layer $\ell$.

**Rationale:** Attaching a loss signal at every layer makes gradient-×-input attribution well-posed because every intermediate node receives direct supervision.

---

## B. GraphPath — Multi-Head Graph Attention

**Reference:** Ma & Wang, 2024 (Bioinformatics)

### Architecture Overview

GraphPath imposes pathway-pathway adjacency (parent-child and sibling relationships) as a fixed mask within a multi-head Graph Attention Network.

### Pathway Adjacency Graph

A symmetric $\{0, 1\}$ adjacency matrix $A \in \{0,1\}^{N \times N}$ is constructed where:

$$A_{pq} = 1 \iff \text{pathways } p \text{ and } q \text{ are Reactome parent-child or share a common parent (siblings)}$$

This yields 4,672 undirected edges with mean node degree 5.5. Self-loops are then added so every node can attend to its own embedding.

### Input Projection

A shared weight matrix $W_{\text{proj}} \in \mathbb{R}^{1 \times d}$ plus pathway-specific bias $b_p \in \mathbb{R}^d$ lift each scalar ssGSEA score to an embedding of dimension $d=64$, with Tanh activation:

$$h^{(0)}_p = \tanh(x_p \cdot W_{\text{proj}} + b_p) \quad (3)$$

### Multi-Head Graph Attention

Embeddings pass through a single multi-head GAT layer with $K=3$ heads and ELU output activation. For each head $k$, per-edge attention coefficients are computed as:

$$\alpha^{(k)}_{ij} = \text{softmax}_{j \in \mathcal{N}(i)} \text{LeakyReLU}_{0.2}\left(a^{(k)\top}\left[W^{(k)}h_i \parallel W^{(k)}h_j\right]\right) \quad (4)$$

Non-edges ($A_{ij}=0$) are masked to $-\infty$ before softmax, and attention dropout ($p=0.4$) is applied during training. The $K$ per-head embeddings are concatenated to dimension $K \cdot d$.

### Output and Classification

A shared $\text{Linear}(Kd \to 1)$ followed by Tanh collapses each node to a scalar readout. One fully-connected layer maps the resulting $N$-vector to three binary head logits. Finally, sigmoid activation and multi-label classification are applied.

---

## C. PATH — Edge-Aware Graph Transformer

**Reference:** Howlader et al., 2026 (arXiv)

### Architecture Overview

PATH replaces the binary adjacency with a weighted graph derived from gene-set overlap, augmenting a Graph Transformer with Laplacian positional encodings and edge-conditioned attention biases.

### Weighted Pathway Adjacency

The pathway-pathway adjacency weight is the Jaccard similarity of Reactome gene memberships:

$$A_{pq} = \frac{|G_p \cap G_q|}{|G_p \cup G_q|}, \quad |G_p| \geq 15 \quad (5)$$

computed from the Reactome GMT and renormalised to $[0, 1]$. Only pathways with at least 15 Reactome member genes qualify as graph nodes; this yields 1,431 nodes, 243,210 weighted edges, and mean degree 339.9.

### Input Projection and Positional Encoding

Because the dataset contains only pathway-level ssGSEA scores (no gene-level CNV or mutation profiles), PATH's original Stage 1 (FiLM-modulated gene embeddings) and Stage 2 (intra-pathway attention pooling) are replaced by the same per-pathway projection used in GraphPath: mapping each scalar score to $\mathbb{R}^d$ (d=64) via shared weight and pathway-specific bias, followed by Tanh.

Laplacian positional encodings are then added: the top-$k=16$ non-trivial eigenvectors of the symmetrically normalised Laplacian $L = I - D^{-1/2}AD^{-1/2}$ are concatenated with normalised node degree $\deg(p)/(N + \epsilon)$. These form a $(k+1)$-dimensional positional feature per node, projected to $\mathbb{R}^d$ via a learnable linear layer. During training, the sign of each eigenvector is randomly flipped independently per epoch to resolve sign ambiguity. Positional embeddings are added to node features before the transformer stack.

### Edge-Aware Transformer Blocks

$L=2$ successive transformer blocks each compute scaled dot-product multi-head self-attention ($H=4$ heads) augmented by two structural biases.

**Soft structural mask:** Non-edges are heavily down-weighted while remaining gradient-reachable:

$$m^{(\text{struct})}_{pq} = \begin{cases} 0 & \text{if } A_{pq} > 0 \text{ or } p = q, \\ -10 & \text{otherwise} \end{cases}$$

This allows the model to rewire the prior graph when data warrants.

**Edge-conditioned attention bias:** A learnable per-layer scalar edge feature is aggregated over attention heads:

$$\phi^{(\ell)}_{pq} = \frac{1}{H} \sum_{h=1}^{H} \log \text{softplus}\left(w^{(\ell)}_h e^{(\ell)}_{pq} + b^{(\ell)}_h\right) + \epsilon \quad (6)$$

where $e^{(\ell)}_{pq}$ is initialised from the Jaccard adjacency weight. The combined mask $m^{(\text{struct})}_{pq} + \phi^{(\ell)}_{pq}$ is added to raw attention logits before softmax.

After attention, a residual-wrapped GELU feed-forward network (expansion factor 4) with per-token Batch Normalisation updates node features. A parallel two-layer MLP with Batch Normalisation updates edge features between blocks.

### Readout and Classification

Node tokens are collapsed to a graph-level embedding via attention-weighted readout:

$$g = \sum_{p} w_p x_p^{(L)}, \quad w_p = \text{softmax}_p\left(v^\top \tanh\left(U x_p^{(L)}\right)\right) \quad (7)$$

where $U \in \mathbb{R}^{d \times (d/2)}$ and $v \in \mathbb{R}^{d/2}$ are learnable.

The graph embedding $g$ passes through a classification head:

$$\text{Linear}(d \to d) \to \text{BN} \to \text{GELU} \to \text{Dropout}(p=0.2) \to \text{Linear}(d \to 3)$$

producing three binary head logits.

---

## D. Loss, Training, and Reproducibility

### Objective

All three models minimise per-head class-weighted BCE on the 3-bit label. Positive-class weights are computed on the training fold as:

$$w_h = \frac{\# \text{neg}_h}{\# \text{pos}_h}$$

and clipped to $[0.1, 20]$. For BINN, loss is applied to the averaged sigmoid outputs; for GraphPath and PATH, loss is applied to raw logits via `F.binary_cross_entropy_with_logits` for numerical stability.

### Optimisers and Schedules

Optimiser choices follow each reference paper:

| Model | Optimiser | Learning Rate | Weight Decay | Batch Size |
|-------|-----------|---------------|--------------|------------|
| **BINN** | Adam | $10^{-3}$ | $10^{-3}$ | 32 |
| **GraphPath** | SGD (momentum 0.9) | $5 \times 10^{-2}$ | $5 \times 10^{-2}$ | 32 |
| **PATH** | AdamW | $10^{-4}$ | $5 \times 10^{-4}$ | 16 |

All three use ReduceLROnPlateau scheduling on validation loss (reduction factor 0.5; plateau patience of 7, 8, and 10 epochs for BINN, GraphPath, and PATH respectively).

### Early Stopping

Early stopping monitors validation loss with patience 20 (BINN) or 25 (GraphPath, PATH). For PATH, a minimum of 25 training epochs is required before the patience counter activates, preventing premature termination during the slow initial convergence of the transformer stack. The checkpoint achieving the lowest validation loss is restored before evaluation.

### Reproducibility

All models are implemented in PyTorch 2.0 with scikit-learn for metrics. Random seeds are fixed identically across Python, NumPy, and PyTorch (seed=42) for all models and cohorts. Every figure in this paper is regenerated by running:

```bash
scripts/run_all.sh inside binn/, graphpath/, and path/,
followed by paper/build.sh
```

---

## Summary: Comparison of Inductive Biases

| Aspect | BINN | GraphPath | PATH |
|--------|------|-----------|------|
| **Graph encoding** | Sparse masked hierarchy | Fixed parent-child + sibling adjacency | Weighted Jaccard similarity |
| **Attention mechanism** | Multi-layer auxiliary heads | Fixed-mask 3-head GAT | Learnable edge-aware 4-head Transformer |
| **Positional structure** | Layer depth | Implicit (implicit node ordering) | Laplacian spectral encoding |
| **Rewiring capability** | None (fixed mask) | None (fixed adjacency) | Yes (soft mask + learnable edges) |
| **Complexity** | Low | Medium | High |

