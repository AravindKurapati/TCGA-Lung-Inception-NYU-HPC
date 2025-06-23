# **TCGA Lung Cancer Classification using InceptionV3 on NYU HPC**


## Overview

This repository implements and extends the methodology from the paper:  
**"Mapping the landscape of histomorphological cancer phenotypes using self-supervised learning on unlabeled, unannotated pathology slides"** (Nature Cancer, 2023).  
The core contribution of the paper is **Histomorphological Phenotype Learning (HPL)** — a self-supervised framework for discovering morphological patterns in whole slide images (WSIs) without requiring expert annotations.

This approach was applied to **lung adenocarcinoma (LUAD)** slides from TCGA to:
- Uncover **meaningful tissue-level phenotypes**
- **Cluster tile-level embeddings** into interpretable groups (HPCs)
- **Correlate phenotypes** with survival, transcriptomic signatures, and cell-type distributions

---

##  Objectives

- Identify distinct **histomorphological patterns** (e.g., acinar, papillary, solid) via unsupervised learning  
- Represent slides as **compositions of phenotypic clusters**  
- Use cluster distributions to:
  - Predict LUAD vs LUSC (Logistic Regression)
  - Predict survival (Cox Regression)  
- Ensure interpretability with SHAP explanations



---

## Dataset

- **TCGA Lung Adenocarcinoma (LUAD) and Lung Squamous Cell Carcinoma (LUSC)**  
- **Whole Slide Images (WSIs)** with >400K tiles extracted  
- Each tile: 224×224 patches sampled from high-resolution slides (20x or 40x)

---

##  HPL Methodology

![image](https://github.com/user-attachments/assets/93e2fed5-95fb-494a-9ee2-b6d7fd26d134)

### Step 1: WSI Preprocessing
- WSIs split into **tiles** (~224×224 or 299×299)  
- Background tiles removed based on tissue mask thresholds  
- **Normalization** applied to correct staining variations

### Step 2: Self-Supervised Representation Learning
- Used **Barlow Twins** to learn 128D embeddings per tile  
- Trained without labels or annotations  
- Captures structure, morphology, and texture

### Step 3: Tile Clustering to Define HPCs
- 200,000 tile embeddings sampled  
- **UMAP** → dimensionality reduction  
- **Leiden clustering** over KNN graph (K=250)  
- Result: **46 clusters (HPCs)** of histomorphological patterns

### Step 4: Slide-Level Aggregation
- For each slide, compute % of tiles assigned to each HPC  
- Represent slides as a vector of HPC proportions (`w0, w1, ..., wC-1`)

---

## Downstream Tasks

### 🔹 LUAD vs LUSC Classification
- Logistic Regression using HPC proportion vectors as input  
- **TCGA 5-fold CV**: AUC = **0.93**  
- **NYU cohort**: AUC = **0.99**

###  Survival Prediction (OS, RFS)
- **Cox Proportional Hazards model** with HPC proportions  
- TCGA C-index = **0.60**, NYU1 = **0.65**
- HPCs enriched with lymphocyte infiltration → better prognosis

---

##  Interpretability

- Used **SHAP values** to measure HPC impact on predictions  
- Visualized positive and negative contributors to hazard ratios and class probabilities  
- Linked clusters to known histological types and immune signatures

---

##  Biological Interpretations

- HPCs matched known LUAD morphologies (e.g., acinar, papillary, lepidic)  
- Positive survival indicators:  
  - High lymphocyte presence  
  - Inflammatory cell signatures (TILs, macrophages)  
- Negative survival indicators:  
  - Proliferation markers  
  - Solid tumor growth  
  - Sparse immune infiltration

---

## Citation


@article{QuirosCoudray2024,
	author = {Claudio Quiros, Adalberto and Coudray, Nicolas and Yeaton, Anna and Yang, Xinyu and Liu, Bojing and Le, Hortense and Chiriboga, Luis and Karimkhan, Afreen and Narula, Navneet and Moore, David A. and Park, Christopher Y. and Pass, Harvey and Moreira, Andre L. and Le Quesne, John and Tsirigos, Aristotelis and Yuan, Ke},
	journal = {Nature Communications},
	number = {1},
	pages = {4596},
	title = {Mapping the landscape of histomorphological cancer phenotypes using self-supervised learning on unannotated pathology slides},
	volume = {15},
	year = {2024}}
}




# How I used the SSL Embeddings and InceptionV3

I implemented a deep learning pipeline for classifying lung cancer subtypes — **LUAD (Lung Adenocarcinoma)** vs. **LUSC (Lung Squamous Cell Carcinoma)** — from histopathology whole-slide images (WSIs) using **InceptionV3** on **NYU's HPC cluster**. It builds upon prior work that used self-supervised learning (SSL) to extract tile-level embeddings.

---

## Project Objective

Extend the HPL pipeline by building a **deep learning model** to directly classify slide-level embeddings into LUAD or LUSC using a modified **InceptionV3** model.

---

## Dataset Overview

- Source: **TCGA LUAD and LUSC datasets**
- Format: Preprocessed tiles in HDF5 format
- Tile Embeddings: **128D vectors** from pretrained **Barlow Twins** SSL model
- Total:
  - **Train Tiles**: 537,474 → 628 slides
  - **Validation Tiles**: 154,240 → 197 slides
  - **Test Tiles**: 149,265 → 197 slides
- Balanced label distribution across LUAD and LUSC

---

## 🔧 Methodology

### Tile Aggregation
- Tiles grouped by slide ID
- Reconstructed into **(224 × 224 × 128)** tensors
- Treated as "images" for CNN input

### InceptionV3 Model (Modified)
- `input_shape=(224, 224, 128)`
- Removed top classification layer (`include_top=False`)
- Added:
  - `GlobalAveragePooling2D`
  - `Dense(512, ReLU)`
  - `Dropout(0.5)`
  - `Dense(1, Sigmoid)`
- `weights=None` (not using ImageNet due to 128 channels)

### Training Setup
| Parameter        | Value                     |
|------------------|---------------------------|
| Optimizer        | SGD with momentum (0.9)   |
| Learning Rate    | 0.01                      |
| Nesterov         | True                      |
| Loss Function    | Binary Crossentropy       |
| Evaluation       | Accuracy, AUC             |
| Hardware         | NYU HPC (multi-GPU setup) |

---

## Results

| Metric   | Value        |
|----------|--------------|
| Accuracy | ~78%         |
| AUC      | ~0.65        |

Model performance shows reasonable classification accuracy but limited discriminative power (lower AUC), possibly due to compactness of SSL embeddings and limited interpretability.

---

## Comparison with HPL Pipeline

| Aspect                | HPL (Prior Work)                          | This Work (Current)                      |
|-----------------------|-------------------------------------------|------------------------------------------|
| Tile Processing       | SSL + Leiden Clustering → HPCs           | SSL embeddings → CNN input (InceptionV3) |
| Classification        | Logistic Regression on HPC proportions   | InceptionV3 Deep Learning Model          |
| Interpretability      | High (via SHAP, pathology-labeled HPCs)  | Low (CNN is a black box)                 |
| Performance (AUC)     | 0.93 (TCGA), 0.99 (NYU)                   | ~0.65                                     |

---

## Learnings

- SSL embeddings offer dimensional efficiency but may lose visual details.
- Deep models struggle without RGB or spatial patterns.
- End-to-end fine-tuning might improve accuracy.
- Interpretability is harder in CNNs compared to cluster-based models.

---

## Future Directions

- Add **attention or MIL-based aggregation** methods
- Fine-tune SSL + classifier jointly (end-to-end)
- Integrate **multi-modal features** (genomics, clinical)

---

## Visual Examples

![transformed_1](https://github.com/user-attachments/assets/d94675b9-2972-46f0-8011-9d070ffe1d2c)


![image](https://github.com/user-attachments/assets/78159674-d593-4560-a988-06878975ae30)


![image](https://github.com/user-attachments/assets/a48cb556-8a26-4b04-b3e6-8e89c2f079e2)

![image](https://github.com/user-attachments/assets/c73b1e1a-ff37-4650-b525-be5b674652b3)


![TCGA-33-4532-01Z-00-DX1_mask](https://github.com/user-attachments/assets/5659c94e-e475-46b0-8d15-e18e501f89af)


![cluster_13_train](https://github.com/user-attachments/assets/4f03045c-a59c-47ee-8a7e-1ecba9514724)




