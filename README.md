# DF-EL++ 🚀

[![Python](https://img.shields.io/badge/Python-3.7%2B-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-1.12%2B-red.svg)](https://pytorch.org/)
[![CUDA](https://img.shields.io/badge/CUDA-11.6%2B-green.svg)](https://developer.nvidia.com/cuda-toolkit)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

> **Fast and Faithful: Scalable Neuro-Symbolic Learning and Reasoning with Differentiable Fuzzy EL++**  
> *KDD 2026*

This repository provides a from-scratch PyTorch implementation of the DF-EL++ framework, designed to be a faithful reproduction of the paper's methodology.


## 📋 Overview

<div align="center">
  <img src="./images/flowchart1.png" alt="DF-EL++ Framework Overview" width="85%">
  <p><em>Figure 1: The DF-EL++ framework operates as a principled refinement engine, using logical constraints to improve the output of neural perception models.</em></p>
</div>

**DF-EL++** is the first end-to-end differentiable framework that unifies PTIME-complete reasoning with neural learning, resolving the persistent trade-off between logical rigor and computational scale in neuro-symbolic AI.

### 🎯 Key Features

-  **Pure Python & PyTorch**: A clean, modern implementation with no external language dependencies
-  **Principled Refinement Engine**: Neural networks produce initial fuzzy knowledge bases, refined through gradient-based optimization
-  **Massive Scale**: Validated on ontologies like SNOMED CT with **377K concepts**
-  **State-of-the-art Performance**: Achieves up to **42% relative improvement** in Hits@1 on KBC tasks

---

## 🛠️ Installation & Setup

### Prerequisites

The framework is built entirely in Python. All experiments were designed to be run on servers equipped with NVIDIA GPUs.

| Component | Version | Purpose |
|:----------|:--------|:--------|
| **Hardware** | NVIDIA RTX 3090+ | Recommended for GPU acceleration |
| **Python** | 3.7.0+ | Core framework language |
| **PyTorch** | 1.12+ | Deep learning backend |
| **CUDA** | 11.6+ | For GPU support |

### Dependencies

```bash
# 1. Clone the repository
git clone https://github.com/your-username/DF-EL-plus-plus.git
cd DF-EL-plus-plus

# 2. Install Python dependencies
pip install -r requirements.txt
```

<details>
<summary>📦 Full Dependency List</summary>

```
torch>=1.12.0
numpy>=1.21.4
pandas>=1.1.3
matplotlib>=3.3.2
scikit-learn>=0.24.0
tqdm>=4.62.0
loguru>=0.6.0
pyparsing>=3.0.6
```

</details>

---

## 🔧 Data Preprocessing

### Normalization of EL++ Ontologies

The input EL++ ontology must first be normalized. Our Python script systematically decomposes complex axioms into elementary "normal forms" (NF1-NF6), preserving the original semantics while enabling stable gradient-based learning.

```bash
# Run the Python-based normalization script
python preprocess.py \
    --input_ontology ontologies/snomed.owl \
    --output_dir data/snomed_normalized
```

### 📁 Output Structure

The script will create a directory containing the processed data, split into training, validation, and test sets.

```
data/snomed_normalized/
├── concepts.txt          # Concept names
├── roles.txt            # Role names
├── individuals.txt      # Individual names
├── train.txt           # 80% of normalized axioms
├── valid.txt           # 10% of normalized axioms
└── test.txt            # 10% of normalized axioms
```

---

## 🚀 Training & Evaluation

### 📊 Knowledge Base Completion (KBC)

This task evaluates the model's ability to predict missing axioms in an ontology.

#### 1. Training

Use `train_kbc.py` to train a model on a preprocessed dataset. The best model checkpoint will be saved based on validation MRR.

```bash
# Example: Training DF-EL++ on SNOMED CT
python train_kbc.py \
    --data_path data/snomed_normalized \
    --embedding_dim 200 \
    --batch_size 512 \
    --save_dir checkpoints/snomed
```

<details>
<summary>🔧 Advanced Training Options</summary>

```bash
python train_kbc.py \
    --data_path data/snomed_normalized \
    --embedding_dim 200 \
    --batch_size 512 \
    --learning_rate 2e-4 \
    --l2_lambda 1e-5 \
    --negative_samples 50 \
    --max_epochs 100 \
    --patience 10 \
    --save_dir checkpoints/snomed
```

</details>

#### 2. Evaluation

Use `evaluate_kbc.py` to evaluate a trained checkpoint on the test set.

```bash
python evaluate_kbc.py \
    --model_path checkpoints/snomed/best_model.pt \
    --data_path data/snomed_normalized
```

### 🖼️ Semantic Image Interpretation (SII)

This task refines noisy outputs from perception models to ensure logical consistency.

#### 1. Training (Refinement)

The `train_sii.py` script simulates the output of a perception model and uses the DF-EL++ framework to refine the noisy beliefs according to an ontology.

```bash
# Example: Refining beliefs for the PASCAL_PART ontology
python train_sii.py \
    --data_path data/pascal_part_normalized
```

#### 2. Evaluation

The `evaluate_sii.py` script assesses the quality of the refined beliefs using classification and logical consistency metrics.

```bash
python evaluate_sii.py \
    --data_path data/pascal_part_normalized
```

---

## ⚙️ Hyperparameter Configuration

The following tables summarize the optimal hyperparameters found in the paper.

### 📈 KBC Task Settings

| Parameter | Tuning Range | **Selected Value (DF-EL++)** | Description |
|:----------|:-------------|:------------------------------|:------------|
| **Optimizer** | - | `Adam` | Adaptive learning rate optimization |
| **Learning Rate** | `{1e-4, 2e-4, 5e-4}` | `2e-4` | Step size for gradient descent |
| **Embedding Dim** | `{100, 200, 400}` | `200` | Dimensionality of embeddings |
| **Batch Size** | `{256, 512, 1024}` | `512` (SNOMED) / `1024` (others) | Samples per iteration |
| **L2 Regularization** | - | `1e-5` | Weight decay coefficient |
| **Negative Samples** | - | `50` | Number of negative samples |
| **Early Stopping** | - | 10 epochs (patience on val MRR) | Prevents overfitting |

### 🎨 SII Task Settings

| Parameter | **Configuration** | Description |
|:----------|:-----------------|:------------|
| **Optimizer** | `Adam` | For refinement optimization |
| **Learning Rate** | `1e-4` | Refinement step size |
| **Max Epochs** | `50` | Per image refinement iterations |
| **Perception Model** | Fast R-CNN (ResNet-50) | Pre-trained backbone |

---

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## ❓ FAQ

<details>
<summary>How to handle OOM errors?</summary>

Reduce the batch size or embedding dimension:
```bash
python train_kbc.py --batch_size 256 --embedding_dim 100
```

</details>

<details>
<summary>Can I use my own ontology?</summary>

Yes! Just ensure it's in OWL format and run:
```bash
python preprocess.py --input_ontology your_file.owl
```

</details>

---

<div align="center">
  <sub>Built with ❤️ for the Neuro-Symbolic AI Community</sub>
</div>
