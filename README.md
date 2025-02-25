# **TCGA Lung Cancer Classification using InceptionV3 on NYU HPC**

## **Overview**
This repository contains a deep learning pipeline for training an **InceptionV3 model** on **TCGA lung cancer histopathology images**. The model classifies images into **LUAD (Lung Adenocarcinoma) or LUSC (Lung Squamous Cell Carcinoma)** using **TensorFlow, HDF5 datasets, and HPC clusters**.  

**⚠ Note:** This code is **specifically designed for NYU's HPC cluster** and requires modifications for use on other systems.

---

## **Project Status**
✅ **Data Processing**: Extracting slide-level images from HDF5 files  
✅ **Patient ID Mapping**: Linking slides to TCGA patient IDs  
✅ **Deep Learning Model**: InceptionV3 trained with **TFRecords**  
✅ **Cluster Analysis**: Identifying high-density tile clusters using **DBSCAN**  
🛠 **Ongoing Work**:
- Enhancing performance with different **augmentation strategies**
- Optimizing computational efficiency on **NYU Greene Cluster**
- Exploring **alternative models like ViTs and EfficientNet**  

---

## **Dataset**
We use the **TCGA lung cancer dataset**, which includes:
- **Histopathology slide images** stored in **HDF5 format**
- **Patient ID labels** for LUAD/LUSC classification
- **Tiling information** to reconstruct whole-slide images

---

## **Methods Implemented**
### **1️⃣ Data Preprocessing**
- **Extract images** from HDF5 files  
- **Normalize pixel intensities** for stable training  
- **Assign labels** to patient slides (LUAD/LUSC)  
- **Split data** into Train, Validation, and Test  

### **2️⃣ Deep Learning Model**
- **Architecture**: InceptionV3 (pretrained weights removed)  
- **Augmentation**: Random flips, rotations, contrast adjustments  
- **Loss Function**: Binary Cross-Entropy  
- **Optimizer**: SGD with momentum  
- **Evaluation**: AUC-ROC and accuracy  

### **3️⃣ HPC-Specific Optimization**
- **TFRecords** to handle large datasets  
- **Multi-GPU training** (NYU HPC)  
- **Batch-wise data loading** for memory efficiency  

### **4️⃣ Tile Clustering & Visualization**
- **DBSCAN clustering** for identifying meaningful regions  
- **Mask verification** to ensure correct label mappings  
- **Slide-wise reconstructions** from tile coordinates  

---

## **Installation**
To install dependencies:
```bash
pip install tensorflow torch torchvision torchaudio 
pip install scikit-learn pandas numpy tqdm biopython h5py matplotlib
