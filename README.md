# Music Genre Classification using Spectrogram Images

This repository presents a comprehensive framework for hierarchical music genre classification using spectrogram images and multiple machine learning and deep learning approaches. It includes implementations with EfficientNet, Audio Spectrogram Transformer (AST), Custom Convolutional Neural Network (CNN), and traditional machine learning models. All models utilize multi-modal spectrogram features (mel, CQT, chroma) and support multi-level genre classification.

---

## Problem Statement

**Goal:**  
To classify music tracks into genres and sub-genres by learning from spectrogram images generated from audio signals. The challenge involves:
- Handling hierarchical genre classification (coarse to fine: classic/non-classic → genre → sub-genre).
- Effectively processing multi-modal spectrogram images.
- Building models that generalize well to unseen music and provide interpretable predictions.

---

## Approach Overview

### 1. **Data Source**
- Uses the [ccmusic-database/music_genre](https://huggingface.co/datasets/ccmusic-database/music_genre) from HuggingFace.
- Each sample contains mel, CQT, and chroma spectrogram images, and three levels of genre labels.

### 2. **Models Implemented**
- **EfficientNet_model.ipynb:** Music genre classification using a pre-trained EfficientNet-B0 backbone.
- **AST_model.ipynb:** Classification using the Audio Spectrogram Transformer (AST), leveraging transformer attention for spectrogram images.
- **CustomCNN_model.ipynb:** Custom CNN with residual and attention blocks, designed for multi-modal spectrogram input.
- **ML_model.ipynb:** Traditional machine learning pipeline with PCA for dimensionality reduction and gradient boosting classifiers.

### 3. **Workflow**
- Data loading & preprocessing
- Multi-level label mapping and encoding
- Model training, validation, and testing
- Advanced techniques: SpecAugment, label smoothing, cosine LR scheduling, mixed precision
- Evaluation with metrics and confusion matrices

---

## Hierarchical Genre Labels

| Level         | Description                         | Example Classes                                  |
|---------------|-------------------------------------|--------------------------------------------------|
| First Level   | Binary                             | Classic, Non-classic                             |
| Second Level  | 9 genres                           | Symphony, Opera, Solo, Chamber, Pop, Dance/House, Indie, Soul/RnB, Rock |
| Third Level   | 16 sub-genres                      | Symphony, Opera, Solo, Chamber, Pop vocal ballad, Adult contemporary, Teen pop, Contemporary dance pop, Dance pop, Classic indie pop, Chamber cabaret and art pop, Soul/RnB, Adult alternative rock, Uplifting anthemic rock, Soft rock, Acoustic pop |

---

## Results Table

| Model                  | Overall Accuracy | Classic/Non-classic | 2nd-level Genres | 3rd-level Sub-genres | Notable Features                       |
|------------------------|:---------------:|:-------------------:|:----------------:|:--------------------:|----------------------------------------|
| **EfficientNet-B0**    |   96.9%         |      100.0%          |     95.9%        |      94.9%           | Transfer learning, multi-input         |
| **AST Transformer**    |   89.3%         |      99.3%          |     86.3%        |      82.3%           | SpecAugment, transformer attention     |
| **Custom CNN**         |   91.1%         |      99.8%          |     90.1%        |      83.5%           | Residual, SE blocks, label smoothing   |
| **ML Model (GBM + PCA)**|  73.3%         |      96.0%          |     65.0%        |      59.0%           | Feature engineering, memory-efficient  |


---

## Repository Structure

| File                       | Description                                                    |
|----------------------------|----------------------------------------------------------------|
| EfficientNet_model.ipynb   | EfficientNet-based deep learning workflow                      |
| AST_model.ipynb            | Audio Spectrogram Transformer (AST) workflow                   |
| CustomCNN_model.ipynb      | Custom CNN architecture for music genre classification         |
| ML_model.ipynb             | Traditional machine learning pipeline                          |
| README.md                  | Project overview and results                                   |

---


