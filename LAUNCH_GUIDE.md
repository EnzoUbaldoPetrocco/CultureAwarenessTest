# CultureAwarenessTest — Script Execution & Launch Guide

This document provides a comprehensive, step-by-step guide to running all execution scripts, training pipelines, dataset analysis tools, and visualization utilities in the **CultureAwarenessTest** project repository.

---

## 1. System Requirements & Setup

### Prerequisites
- **Python**: 3.8, 3.9, or 3.10
- **TensorFlow**: 2.10.1 (with GPU support)
- **CUDA / cuDNN**: Compatible GPU drivers (NVIDIA GPU with min 6GB VRAM recommended)
- **OS**: Windows / Linux

### Environment Installation

1. **Clone & Navigate to Workspace**:
   ```bash
   git clone https://github.com/EnzoUbaldoPetrocco/CultureAwarenessTest.git
   cd CultureAwarenessTest
   ```

2. **Create Virtual Environment**:
   ```bash
   python -m venv .venv
   # Windows PowerShell:
   .venv\Scripts\Activate.ps1
   # Linux/macOS:
   source .venv/bin/activate
   ```

3. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

   *If running experiments under `Mitigated/`:*
   ```bash
   pip install -r Mitigated/requirements.txt
   ```

---

## 2. Main Entry Points & Script Launchers

| Script Path | Purpose | Key Outputs |
| :--- | :--- | :--- |
| [`Mitigated/launch.py`](file:///C:/Users/Utente/Desktop/CultureAwarenessTest/Mitigated/launch.py) | **Primary Benchmark Launcher**: Trains standard control vs. bias-mitigated ResNet models across datasets & cultures. | Checkpoints, confusion matrices, metrics logs in `./try3/` |
| [`Mitigated/launch_shallow.py`](file:///C:/Users/Utente/Desktop/CultureAwarenessTest/Mitigated/launch_shallow.py) | **Shallow Baseline Launcher**: Fits SVM and Random Forest classifiers on flattened grayscale images. | Shallow model evaluation CSVs & confusion matrices |
| [`DatasetAnalysis/examinate.py`](file:///C:/Users/Utente/Desktop/CultureAwarenessTest/DatasetAnalysis/examinate.py) | **Embedding & Distance Metrics**: Extracts ResNet50 embeddings, computes intra/inter-class distances & silhouette scores. | `results_lamp_0.json`, `results_lamp_1.json` |
| [`DatasetAnalysis/k-means.py`](file:///C:/Users/Utente/Desktop/CultureAwarenessTest/DatasetAnalysis/k-means.py) | **Unsupervised Clustering**: Fits K-Means ($K=6$) on image features to evaluate natural cultural clustering. | Console output of predicted vs. true labels |
| [`GradCam/launch.py`](file:///C:/Users/Utente/Desktop/CultureAwarenessTest/GradCam/launch.py) | **GradCAM Visual Explanations**: Generates attention heatmaps across confusion matrix buckets (TP, TN, FP, FN). | Heatmap PNGs in `./TNOAUG/CULTURE<id>/` |
| [`LaunchFiles/diffusion_step_plot.py`](file:///C:/Users/Utente/Desktop/CultureAwarenessTest/LaunchFiles/diffusion_step_plot.py) | **Diffusion Step Plotter**: Generates and plots synthetic minority culture images step-by-step using DDPM. | Sample images in `./Diff_step_plot/` |
| [`LaunchFiles/plot_standard_augmentation.py`](file:///C:/Users/Utente/Desktop/CultureAwarenessTest/LaunchFiles/plot_standard_augmentation.py) | **Classical Augmentation Plotter**: Plots classical augmentation variations across noise levels. | Data augmentation preview figures |
| [`OverallPipeline.ipynb`](file:///C:/Users/Utente/Desktop/CultureAwarenessTest/OverallPipeline.ipynb) | **Interactive Analysis Notebook**: Full end-to-end evaluation, visualization, and paper plot generation. | Interactive plots, summary tables, LaTeX code |

---

## 3. Step-by-Step Execution Instructions

### A. Running Main Bias Mitigation Experiments (`Mitigated/launch.py`)

The main entry point for deep learning mitigation experiments is `Mitigated/launch.py`.

```bash
cd Mitigated
python launch.py
```

#### Configuration Tuple Parameters
Inside `launch.py`, experiments are defined by the `todo_configs` list of 7-element tuples:
```python
(standard, lamp, culture, diffusion, only_min, parify, augment)
```

- **`standard`** (int): `1` = Standard Control Model (Standard ResNet50V2), `0` = Bias-Mitigated Model (Custom variance regularization).
- **`lamp`** (int): `1` = Lamps Dataset, `0` = Carpets Dataset.
- **`culture`** (int): Target majority culture ID (`0`, `1`, or `2`).
- **`diffusion`** (int): `1` = Generate synthetic data via DDPM diffusion model, `0` = Disabled.
- **`only_min`** (int): `1` = Apply diffusion generation ONLY to minority cultures, `0` = All cultures.
- **`parify`** (int): `1` = Enable culture-balanced batch sampling, `0` = Standard sampling.
- **`augment`** (int): `1` = Enable classical data augmentation (rotations, brightness, noise), `0` = Disabled.

---

### B. Running Shallow Machine Learning Baselines (`Mitigated/launch_shallow.py`)

To evaluate traditional shallow models (SVM, Random Forest) on flattened grayscale representations:

```bash
cd Mitigated
python launch_shallow.py
```

---

### C. Running Dataset Quality & Embedding Analysis (`DatasetAnalysis/examinate.py`)

To analyze the dataset feature space, extract ResNet50 embeddings, and measure inter-class / intra-class distance metrics:

```bash
cd DatasetAnalysis
python examinate.py
```
*Outputs:* `results_lamp_1.json` containing mean intra-class distance, mean inter-class distance, separation ratio, and global silhouette score.

To test unsupervised K-Means clustering:
```bash
python k-means.py
```

---

### D. Generating GradCAM Visual Attention Heatmaps (`GradCam/launch.py`)

To produce Gradient-weighted Class Activation Maps (GradCAM) highlighting model focus regions on true/false positive/negative samples:

```bash
cd GradCam
python launch.py
```
*Outputs:* Heatmap overlays saved under `./TNOAUG/CULTURE<0|1|2>/<TP|TN|FP|FN>/`.

---

### E. Plotting Diffusion Synthetic Generation (`LaunchFiles/diffusion_step_plot.py`)

To visualize denoising diffusion steps during synthetic image synthesis:

```bash
cd LaunchFiles
python diffusion_step_plot.py
```

---

## 4. Programmatic Pipeline Usage (`ProcessingClass`)

You can also orchestrate custom experiments programmatically using `ProcessingClass`:

```python
from Processing.processing import ProcessingClass

# 1. Initialize Pipeline Processor
procObj = ProcessingClass(
    shallow=0,          # 0 = Deep Learning (ResNet), 1 = Shallow (SVM)
    lamp=1,             # 1 = Lamps Dataset, 0 = Carpets Dataset
    gpu=True,           # Enable GPU execution
    memory_limit=6000,  # GPU VRAM limit in MB
    basePath="./results/"
)

# 2. Train Model
procObj.process(
    standard=0,         # 0 = Mitigated Model, 1 = Standard Model
    culture=0,          # Culture 0 as majority culture
    percent=0.05,       # Retain 5% minority data ratio
    augment=1,          # Classical augmentation enabled
    diffusion=1,        # Diffusion augmentation enabled
    only_minority_diffusion=1,
    parify_batches_diffusion=1
)

# 3. Test & Log Metrics
procObj.test(standard=0, culture=0)
```

---

## 5. Hardware Optimization & Troubleshooting

### GPU Memory Management
If you encounter TensorFlow GPU Out-of-Memory (OOM) errors:
1. Reduce `memory_limit` in the script constructor (e.g., from `6000` to `4000`).
2. Reduce batch size `bs` in `procObj.process(batch_size=16)`.
3. Set CPU fallback mode: `gpu=False` in `ProcessingClass`.

### Dataset Directory Placement
Ensure dataset image directories follow the expected folder paths defined in [`Utils/Data/deep_paths.py`](file:///C:/Users/Utente/Desktop/CultureAwarenessTest/Utils/Data/deep_paths.py) and [`Utils/Data/shallow_paths.py`](file:///C:/Users/Utente/Desktop/CultureAwarenessTest/Utils/Data/shallow_paths.py).
