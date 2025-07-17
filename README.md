# GeMM-GAN

**GeMM-GAN: A Multimodal Generative Model Conditioned on Histopathology Images and Clinical Descriptions for Gene Expression Profile Generation**

This repository contains the official PyTorch implementation of GeMM-GAN, a multimodal WGAN-GP framework that generates realistic gene expression profiles conditioned on histopathology image patches and clinical descriptions.

📄 **Paper**: _Accepted at ICIAP 2025_  
👩‍💻 **Authors**: Francesca Pia Panaccione, Carlo Sgaravatti, Pietro Pinoli  
📚 **Dataset**: TCGA – The Cancer Genome Atlas  
📦 **Frameworks**: PyTorch, Hugging Face Transformers

---

## 🧠 Overview

GeMM-GAN leverages:
- A **pretrained Vision Transformer (UNI)** for histopathology image patches
- A **clinical language model (Clinical ModernBERT)** for patient metadata
- A **Multimodal Fusion module** with FiLM and Cross-Attention
- A **Wasserstein GAN with Gradient Penalty (WGAN-GP)** for transcriptomic profile generation


![Model Architecture](ICIAP_Architecture.pdf)

