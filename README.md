# BERT Model Fine-Tuning Project

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.12+-blue.svg" alt="Python Version">
  <img src="https://img.shields.io/badge/PyTorch-2.0+-red.svg" alt="PyTorch Version">
  <img src="https://img.shields.io/badge/Transformers-4.0+-green.svg" alt="Transformers Version">
  <img src="https://img.shields.io/badge/License-MIT-yellow.svg" alt="License">
</p>

> 🚀 Efficient BERT fine-tuning using LoRA, Prefix-Tuning, and Knowledge Distillation techniques

## 📋 Table of Contents

- [Overview](#-overview)
- [Features](#-features)
- [Project Structure](#-project-structure)
- [Quick Start](#-quick-start)
- [Installation](#-installation)
- [Fine-Tuning Techniques](#-fine-tuning-techniques)
  - [LoRA (Low-Rank Adaptation)](#lora-low-rank-adaptation)
  - [Prefix-Tuning](#prefix-tuning)
  - [SVD-LoRA](#svd-lora)
- [Training Results](#-training-results)
- [Notebooks](#-notebooks)
- [Docker Support](#-docker-support)
- [Citation](#-citation)

---

## 📖 Overview

This project implements various efficient fine-tuning methods for BERT models, including:

- **LoRA (Low-Rank Adaptation)**: Reduces trainable parameters by decomposing weight updates into low-rank matrices
- **Prefix-Tuning**: Adds learnable continuous vectors to the input
- **SVD-LoRA**: An innovative approach combining SVD decomposition with LoRA
- **Knowledge Distillation**: Transfer knowledge from teacher to student models

## ✨ Features

| Feature | Description |
|---------|-------------|
| 🔥 LoRA | Low-rank adaptation with minimal parameter overhead |
| 📝 Prefix-Tuning | Continuous prefix vectors for task-specific prompts |
| 🎯 SVD-LoRA | SVD-based decomposition for better interpretability |
| ⚡ Mixed Precision | AMP training for faster computation |
| 📦 Gradient Accumulation | Support for larger effective batch sizes |
| 🐳 Docker | Containerized development environment |

---

## 📂 Project Structure

```
Bert_model/
├── Jupyter/                      # Jupyter notebooks
│   ├── bert.ipynb               # BERT base model examples
│   ├── distill.ipynb            # Knowledge distillation & SVD-LoRA
│   └── mood_analyse.ipynb       # Sentiment analysis examples
├── docker/                      # Docker configuration
│   └── Dockerfile
├── Transformer+NLP+Code/       # Learning materials
│   └── chapter1-12.py          # Tutorial code
├── test_gpu.py                  # GPU test script
├── requirements.txt            # Python dependencies
└── README.md                   # This file
```

---

## 🚀 Quick Start

```bash
# Clone the repository
git clone https://github.com/wonder-prince/Bert_model.git
cd Bert_model

# Create virtual environment
python -m venv venv
source venv/bin/activate  # Linux/Mac
# venv\Scripts\activate   # Windows

# Install dependencies
pip install -r requirements.txt
```

---

## 📦 Installation

### Prerequisites

- Python 3.12+
- CUDA 12.4+ (for GPU training)

### Install PyTorch

```bash
pip install torch==2.6.0 torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124
```

### Install Dependencies

```bash
pip install -r requirements.txt
```

### Environment Variables

For users in China, use Hugging Face mirror:

```bash
export HF_ENDPOINT=https://hf-mirror.com
```

Or in Python:

```python
import os
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
```

---

## 🔧 Fine-Tuning Techniques

### LoRA (Low-Rank Adaptation)

LoRA reduces the computational cost of fine-tuning by decomposing the weight update matrix:

```
ΔW = BA
```

Where:
- `B ∈ ℝ^(d×r)`: Down projection matrix
- `A ∈ ℝ^(r×k)`: Up projection matrix  
- `r << min(d,k)`: Rank (typically 8-16)

**Benefits:**
- Only ~0.1% of original parameters need training
- No inference latency
- Can be easily merged with base model

### Prefix-Tuning

Prefix-Tuning adds a learnable continuous vector sequence before the input:

```
Prefix: [p₁, p₂, ..., pₙ] + Input: [x₁, x₂, ..., xₘ]
```

**Benefits:**
- Non-invasive to model architecture
- Task-specific prefix can be stored separately
- Works well with limited data

### SVD-LoRA

An innovative approach combining SVD decomposition with LoRA:

```
ΔW = U · diag(S) · V
```

Where:
- `U ∈ ℝ^(d×r)`: Left singular vectors
- `S ∈ ℝ^r`: Singular values (interpretable)
- `V ∈ ℝ^(r×k)`: Right singular values

**Benefits:**
- Better interpretability through singular values
- Supports adaptive rank allocation (like AdaLoRA)
- More efficient parameter usage

---

## 📊 Training Results

### SVD-LoRA + Prefix-Tuning

```
TrainOutput(
    global_step=32940,
    training_loss=0.9514,
    metrics={
        'train_runtime': 47886.82s,
        'train_samples_per_second': 5.503,
        'train_steps_per_second': 0.688,
        'total_flos': 5.16e+16,
        'train_loss': 0.9514,
        'epoch': 2.0
    }
)
```

### Parameter Efficiency

| Method | Trainable Parameters | Reduction |
|--------|---------------------|-----------|
| Full Fine-tuning | 109M (100%) | - |
| LoRA | ~0.3M (0.3%) | 99.7% |
| Prefix-Tuning | ~0.08M (0.07%) | 99.93% |
| SVD-LoRA | ~0.2M (0.2%) | 99.8% |

---

## 📓 Notebooks

| Notebook | Description |
|----------|-------------|
| `bert.ipynb` | Basic BERT usage and fine-tuning examples |
| `distill.ipynb` | Knowledge distillation, LoRA, SVD-LoRA implementation |
| `mood_analyse.ipynb` | Sentiment analysis with fine-tuned models |

---

## 🐳 Docker Support

Build and run with Docker:

```bash
cd docker
docker build -t bert-model .
docker run --gpus all -it bert-model
```

---

## 📚 Learning Materials

The `Transformer+NLP+Code/` directory contains chapter-by-chapter tutorials:

- `chapter1-8.py`: Transformer basics
- `chapter9-12.py`: Advanced NLP techniques

---

## 📄 License

MIT License

---

## 🤝 Citation

If you use this code, please cite:

```bibtex
@misc{Bert_model,
  author = {wonder-prince},
  title = {BERT Model Fine-Tuning with Efficient Methods},
  year = {2024},
  publisher = {GitHub},
  url = {https://github.com/wonder-prince/Bert_model}
}
```

---

## 🙏 Acknowledgments

- [Hugging Face Transformers](https://github.com/huggingface/transformers)
- [LoRA Paper](https://arxiv.org/pdf/2106.09685)
- [Prefix-Tuning Paper](https://arxiv.org/abs/2101.00190)
