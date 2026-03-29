# Shrink or Sink: STL-10 Model Compression Challenge

This repository contains the source code and final model for the Shrink or Sink competition.

## 🚀 Results
- **Model Size**: ~1.38 MB (INT8 Quantized TFLite)
- **Test Accuracy**: Improved (87.35%+ via Boosted V2)
- **Compression Ratio**: ~95x vs ResNet-18 baseline

## 🧠 Model Architecture
We use a **MobileNetV3-Small** student with a **boosted `alpha=1.25`** multiplier for higher capacity. The model is trained using a **Boosted V2 Knowledge Distillation** pipeline (Noisy Student) with dynamic temperature scheduling to maximize information transfer from the teacher.

## 🛠️ Pipeline Details (6 Phases)
To achieve high accuracy with a tiny model, we implement a robust 6-phase pipeline:
1.  **Teacher Initialization**: A ResNet-18 model is trained on the 5,000 labeled images.
2.  **Pseudo-labeling**: The teacher generates soft labels for 100,000 unlabeled images. We add an **11th "Other" class** to safely map low-confidence samples.
3.  **Noisy Student Phase**: The ResNet-18 is re-trained from scratch on the combined 105,000 images (Pseudo + Labeled) to build a noise-resistant teacher.
4.  **Clean Re-calibration**: The Noisy Student teacher is fine-tuned exclusively on the 5,000 labeled images with a very low learning rate to "re-center" on clean data.
5.  **Boosted Distillation**: The knowledge from the calibrated teacher is distilled into a **MobileNetV3-Small (alpha=1.25)** student using **Dynamic Temperature Scheduling** (5.0 -> 1.0).
6.  **Quantization**: The student model is converted to **INT8 TFLite** using Post-Training Quantization (PTQ) with a representative dataset subset.

## 📦 Reproducibility
To reproduce the training and evaluation:

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Run Training
The `train.py` script is a **complete one-click reproducibility pipeline** that executes all 6 phases (Initialization, Pseudo-labeling, Noisy Student, Calibration, Distillation, and Quantization) sequentially.
```bash
python train.py --data_dir /path/to/stl10_binary
```
> [!IMPORTANT]
> This script performs extensive training on over 105,000 images and may take significant time (10-20+ hours depending on GPU).

### 3. Run Testing
```bash
python test.py --data_dir /path/to/stl10_binary --model_path model.tflite
```

## 📂 File Structure
- `model.py`: MobileNetV3 architecture definition.
- `train.py`: Unified training and quantization pipeline.
- `test.py`: Evaluation script supporting .h5 and .tflite formats.
- `model.tflite`: Final submitted model.
