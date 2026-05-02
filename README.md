# EmoNeXt Optimized for FER2013

This repository is an optimized implementation of the **EmoNeXt** architecture for Facial Emotion Recognition (FER). It features structural fixes and training enhancements that improve stability and performance on the FER2013 dataset.

## Key Improvements

### 1. Model Architecture (`models.py`)
* **Attention Scaling:** Fixed the scale factor in `DotProductSelfAttention` to prevent gradient vanishing, correcting the original implementation.
* **Spatial Feature Retention:** Modified `forward_features` to keep spatial dimensions (B, C, 7, 7) instead of applying early global pooling.
* **Token-based Attention:** Updated `forward` to perform attention on 49 spatial tokens, allowing the model to learn relationships between facial regions like eyes and mouth.
* **Self-Attention Loss:** Corrected the SA Loss calculation to compute the mean per-row for proper row-wise comparison and broadcasting.
* **Loss Balancing:** Introduced lambda=0.1 for SA Loss and added **Label Smoothing (0.1)** to better handle the noisy labels in FER2013.

### 2. Training Pipeline (`train.py`)
* **Scheduler Upgrade:** Replaced warm restarts with a **Linear Warmup (10 epochs) + Cosine Decay** schedule for improved fine-tuning stability.
* **Validation Fixes:** * Increased validation `batch_size` to 32 (from 1) for significantly faster evaluation.
* Changed `RandomCrop` to `CenterCrop` in validation to ensure deterministic and consistent results.
* **Optimization:** Increased Early Stopping patience to **20** and updated to the latest `torch.amp.GradScaler` API.

## Performance Results

The model achieves highly competitive results on the **FER2013** dataset:

| Metric | Value |
| :--- | :--- |
| **Test Accuracy (EMA)** | **72.51%** |
| **Val Accuracy (Best)** | **73.07%** |
| EmoNeXt-Tiny (Original Paper) | 73.34% |

### Per-class Accuracy (EMA):
* **Happy:** 89.7% | **Surprise:** 84.1% | **Neutral:** 71.2% | **Disgust:** 71.4%
* **Angry:** 65.1% | **Sad:** 61.5% | **Fear:** 55.3%

## Quick Start
1. [Install CUDA](https://developer.nvidia.com/cuda-downloads)

2. [Install PyTorch 1.13 or later](https://pytorch.org/get-started/locally/)

3. **Install Dependencies:**
   ```bash
   pip install -r requirements.txt
4. **Run Training:**
        python train.py --dataset-path='FER2013' 
                        --batch-size=32 
                        --gradient-accumulation-steps=2 
                        --lr=3e-05 --epochs=300 
                        --amp 
                        --in_22k 
                        --num-workers=0 
                        --model-size='tiny'
## Acknowledgments
Original EmoNeXt paper: "EmoNeXt: an Adapted ConvNeXt for facial Emotion Recognition". This codebase builds upon the original implementation.   