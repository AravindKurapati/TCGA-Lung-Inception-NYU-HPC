# TCGA Lung Cancer Classification using InceptionV3 on NYU HPC

## Overview

This repository implements a deep learning pipeline for classifying lung cancer subtypes — **LUAD (Lung Adenocarcinoma)** vs **LUSC (Lung Squamous Cell Carcinoma)** — from histopathology whole-slide images (WSIs) using a modified **InceptionV3** model trained on self-supervised tile embeddings, running on NYU's HPC cluster.

It builds upon the **Histomorphological Phenotype Learning (HPL)** framework introduced in:

> Quiros et al., *"Mapping the landscape of histomorphological cancer phenotypes using self-supervised learning on unannotated pathology slides"*, Nature Communications, 2024.

The key difference: the HPL paper discovers unsupervised tissue phenotypes via clustering and uses logistic regression on those cluster proportions for classification. **This work extends that pipeline by training InceptionV3 directly on the SSL tile embeddings**, testing whether a deep neural network can outperform the interpretable cluster-based approach.

---

## Dataset Overview

- **Source**: TCGA LUAD and LUSC datasets
- **Format**: Preprocessed tiles in HDF5 format
- **Tile Embeddings**: **128D vectors** from pretrained **Barlow Twins** SSL model
- **Total**:
  - **Train Tiles**: 537,474 -> 628 slides
  - **Validation Tiles**: 154,240 -> 197 slides
  - **Test Tiles**: 149,265 -> 197 slides
- Balanced label distribution across LUAD and LUSC 

---

## HPL Methodology (Prior Work)

<img width="747" height="937" alt="image" src="https://github.com/user-attachments/assets/4abffc5d-fd68-426d-a59d-2f3f43d0c259" />

The HPL framework operates fully without labels. Below is the pipeline this work reproduces and builds upon.

### Step 1: WSI Preprocessing
WSIs are split into 224×224 tiles. Background tiles (glass, whitespace) are removed using tissue masks. Stain normalization is applied to correct for variation in H&E staining across different labs and scanners.

[![Tissue Mask](images/TCGA-33-4532-01Z-00-DX1_mask.png)](images/TCGA-33-4532-01Z-00-DX1_mask.png)

### Step 2: Self-Supervised Representation Learning
**Barlow Twins** is trained without any labels on the full tile set, using InceptionV3 as the backbone. Two augmented views of each tile are passed through the network and the loss minimizes redundancy across embedding dimensions while keeping same-tile representations consistent. The result is a **128D embedding per tile** capturing morphological structure, texture and cell-level patterns without a single label.

### Step 3: Tile Clustering to Define HPCs
200,000 tile embeddings are sampled, reduced to 2D using **UMAP**, and clustered via **Leiden** over a KNN graph (K=250). This produces **46 Histomorphological Phenotype Clusters (HPCs)**, each representing a distinct tissue pattern such as acinar structures, papillary growth or lymphocyte infiltration.

[![Cluster 13 tiles](images/cluster_13_train.jpg)](images/cluster_13_train.jpg)

### Step 4: Slide-Level Aggregation
For each slide, the proportion of its tiles assigned to each HPC is computed. Every slide becomes a vector of 46 HPC proportions — a structured, biology-grounded fingerprint of its tissue composition.

### Step 5: Downstream Tasks (HPL Paper Results)

**LUAD vs LUSC Classification**
Logistic Regression on HPC proportion vectors:
- TCGA 5-fold CV: AUC = **0.93**
- NYU cohort: AUC = **0.99**

**Survival Prediction**
Cox Proportional Hazards model using HPC proportions:
- TCGA C-index = **0.60**, NYU1 = **0.65**

**Interpretability via SHAP**
SHAP values identify which HPCs drive predictions most, connecting model outputs back to real biology:
- **Positive survival indicators**: HPCs enriched with lymphocyte infiltration (TILs, macrophages) and inflammatory cell signatures — high immune presence correlates with better prognosis
- **Negative survival indicators**: HPCs with proliferation markers, solid tumor growth patterns, and sparse immune infiltration — these correlate with worse outcomes
- Known LUAD morphologies (acinar, papillary, lepidic growth patterns) map cleanly onto specific HPCs, validating that the SSL model learned real tissue biology rather than noise

---

## My Contribution: InceptionV3 Classifier on SSL Embeddings

Rather than using HPC proportion vectors as slide features, this project feeds the raw 128D tile embeddings directly into a modified InceptionV3, treating each slide as a **(224×224×128)** tensor and asking whether a deep model can learn to classify without the clustering intermediate step.

> **Note on input size**: InceptionV3's canonical input is 299×299 but since `weights=None` is used (ImageNet pretraining is incompatible with 128 channels), the input shape is overridden to 224×224 to match the tile extraction size used throughout the HPL pipeline.

[![Transformed input tiles](images/transformed_1.png)](images/transformed_1.png)

### Tile Aggregation
- Tiles grouped by slide ID
- Reconstructed into **(224 × 224 × 128)** tensors -  128 embedding dimensions treated as channels
- Each tensor represents one full slide as a single CNN input

### InceptionV3 Architecture (Modified)

| Layer | Config |
|-------|--------|
| Input shape | (224, 224, 128) |
| Base | InceptionV3, `include_top=False`, `weights=None` |
| Pooling | GlobalAveragePooling2D |
| Dense | 512 units, ReLU |
| Regularization | Dropout(0.5) |
| Output | Dense(1, Sigmoid) |

### Training Setup

| Parameter | Value |
|-----------|-------|
| Optimizer | SGD with momentum (0.9), Nesterov |
| Learning Rate | 0.01 |
| Loss | Binary Crossentropy |
| Evaluation | Accuracy, AUC |
| Hardware | NYU HPC, multi-GPU |

---

## Results & Comparison with HPL

| Aspect | HPL (Prior Work) | This Work |
|--------|-----------------|-----------|
| Tile Processing | SSL + Leiden Clustering → HPCs | SSL embeddings → CNN input |
| Slide Representation | HPC proportion vector (46-dim) | (224×224×128) embedding tensor |
| Classifier | Logistic Regression | InceptionV3 |
| Interpretability | High (SHAP + pathology-labeled HPCs) | Low (CNN black box) |
| AUC | 0.93 (TCGA), 0.99 (NYU) | ~0.65 |
| Accuracy | — | ~78% |

The simpler cluster-based approach outperforms the deep model. The HPL pipeline uses structured, biology-grounded features that logistic regression can exploit cleanly. InceptionV3 trains from scratch here on embedding tensors that don't carry the spatial structure CNNs are designed to exploit and without ImageNet pretraining there is no useful weight initialization. End-to-end fine tuning of the SSL backbone together with the classifier, rather than using frozen embeddings would likely close this gap significantly.

---

## Learnings

- SSL embeddings encode rich morphological information but lose the spatial structure that CNNs exploit — explaining why logistic regression on cluster proportions outperformed the deep model
- Class imbalance at tile level significantly affects convergence; upsampling the minority class was necessary for stable training
- HDF5-based tile storage was essential for efficient random-access I/O at 850K+ tile scale on HPC
- Custom SLURM job scripting on shared infrastructure requires careful GPU and memory allocation to avoid preemption mid-training
- Reproducing the HPL paper's SSL embeddings before building the classifier was critical. Without validated representations any downstream result would be meaningless


## Citation

```bibtex
@article{QuirosCoudray2024,
  author  = {Claudio Quiros, Adalberto and Coudray, Nicolas and Yeaton, Anna and Yang, Xinyu and Liu, Bojing and Le, Hortense and Chiriboga, Luis and Karimkhan, Afreen and Narula, Navneet and Moore, David A. and Park, Christopher Y. and Pass, Harvey and Moreira, Andre L. and Le Quesne, John and Tsirigos, Aristotelis and Yuan, Ke},
  journal = {Nature Communications},
  number  = {1},
  pages   = {4596},
  title   = {Mapping the landscape of histomorphological cancer phenotypes using self-supervised learning on unannotated pathology slides},
  volume  = {15},
  year    = {2024}
}
```
