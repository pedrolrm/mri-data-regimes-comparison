# MRI Data Regimes Comparison

Cientific benchmark and comparative study between **Hybrid Transfer Learning (RadImageNet ResNet-50 + SVM)** and **End-to-End Convolutional Fine-Tuning** for Alzheimer's disease brain MRI classification across multiple data regimes ($N \in \{1{,}000, 5{,}000, 10{,}000, 20{,}000\}$).

---

## 📁 Project Structure

```text
mri-data-regimes-comparison/
├── outputs/
│   ├── figures_full_unfreeze/      # Publication figures for ResNet-50 (Performance, Time, Confusion Matrix)
│   └── figures_svm/                # Publication figures for SVM (Performance, Time, Confusion Matrix)
├── src/
│   ├── eval/                       # Evaluation and inference routines
│   │   ├── __init__.py
│   │   └── eval_svm.py
│   ├── features/                   # Latent space representation extraction
│   │   ├── __init__.py
│   │   └── extract_features.py
│   ├── models/                     # Deep learning backbones and statistical classifiers
│   │   ├── __init__.py
│   │   ├── encoder.py              # Frozen RadImageNet ResNet-50 feature extractor
│   │   ├── resnet50.py             # End-to-end ResNet-50 classifier architecture
│   │   └── svm.py                  # RBF Support Vector Machine implementation
│   ├── plots/                      # Publication-quality plotting and visualization (300 DPI)
│   │   ├── __init__.py
│   │   ├── plot_full_unfreeze.py   # ResNet-50 metric curves, latency, and confusion matrix
│   │   ├── plot_resnet.py          # ResNet-50 baseline visualizations
│   │   ├── plot_svm.py             # SVM performance, latency trend, and confusion matrix
│   │   └── plot_experiments.py    # Cross-architecture comparison plots
│   ├── train/                      # Training pipelines for all paradigms across data regimes
│   │   ├── __init__.py
│   │   ├── train_binary.py         # Two-phase warmup and differential fine-tuning
│   │   ├── train_full_unfreeze.py  # Full unfreezing fine-tuning pipeline
│   │   ├── train_resnet.py         # Multi-class and standard training routines
│   │   └── train_svm_binary.py     # Stratified SVM training across regimes (1k, 5k, 10k, 20k)
│   └── __init__.py
├── tests/
│   └── test_encoder_svm.py         # Automated unit and integration test suite
├── utils/
│   ├── __init__.py
│   └── preprocess_mri.py           # Isotropic resizing with dynamic border color padding
├── DOCUMENTACAO.md                  # Comprehensive technical documentation
├── requirements.txt                # Python dependencies
├── LICENSE                         # Project license
└── .gitignore                      # Git ignore patterns
```

---

## 🚀 Quickstart & Usage

### 1. Feature Extraction (RadImageNet Latent Space)
Extract 2048-dimensional embeddings from skull-stripped MRI slices:
```bash
python src/features/extract_features.py --data_dir data/data_extreme --output_dir data/features_extreme
```

### 2. SVM Training across Data Regimes
Train the RBF Support Vector Machine across strictly stratified sample subsets ($N = 1\text{k}, 5\text{k}, 10\text{k}, 20\text{k}$):
```bash
python src/train/train_svm_binary.py --features_path data/features_extreme/features.npy --labels_path data/features_extreme/labels.npy
```

### 3. End-to-End ResNet-50 Training
Run convolutional fine-tuning across data regimes:
```bash
python src/train/train_full_unfreeze.py --task moderate --all_regimes
```

### 4. Scientific Figure Generation
Generate high-resolution (300 DPI) publication-ready figures in English:
```bash
# Generate ResNet-50 figures (Performance, Latency with scaling trend, Confusion Matrix)
python src/plots/plot_full_unfreeze.py

# Generate SVM figures
python src/plots/plot_svm.py
```

### 5. Running Tests
Execute the automated test suite:
```bash
python -m unittest discover -s tests
```