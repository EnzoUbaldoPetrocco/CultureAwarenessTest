# CultureAwarenessTest

CultureAwarenessTest is a machine learning research framework designed to investigate, measure, and mitigate cultural bias in deep learning models for image classification. The repository contains implementation scripts for training baseline control models (e.g., standard ResNet), bias-mitigated architectures using custom variance-minimization regularization, diffusion-based synthetic data generation (DDPM) for minority culture augmentation, adversarial testing, and model explainability via GradCAM.

---

## Key Features

- **Bias Detection**: Binary and multiclass discriminators designed to evaluate if cultural patterns leak into latent representations.
- **Custom Bias Mitigation**: Regularized objective function designed to minimize the variance of weights and gradients across cultural groups.
- **Synthetic Augmentation**: Denoising Diffusion Probabilistic Models (DDPM) to synthesize representative samples for underrepresented cultures.
- **Adversarial Robustness Evaluator**: Tests resilience under culture-specific Projected Gradient Descent (PGD) perturbations.
- **Explainability (XAI)**: Visualizes network focus areas using Gradient-weighted Class Activation Maps (GradCAM).

---

## Installation & Environment Setup

This project requires a Python environment configured with GPU-enabled TensorFlow. You can set it up using either **Conda** (recommended) or **Pip**.

### Option A: Setup using Conda (Recommended)
An environment configuration file is provided at [`Mitigated/environment.yml`](file:///C:/Users/Utente/Desktop/CultureAwarenessTest/Mitigated/environment.yml). This will install Python 3.9, CUDA Toolkit 11.2, cuDNN 8.1, and all dependencies:

```bash
# Create the environment from the environment.yml file
conda env create -f Mitigated/environment.yml

# Activate the new environment
conda activate petrocco
```

### Option B: Setup using Pip
If you prefer using `pip` inside a virtual environment, ensure you have Python 3.8+ and TensorFlow-compatible CUDA drivers installed. Then, use the requirements file at [`Mitigated/requirements.txt`](file:///C:/Users/Utente/Desktop/CultureAwarenessTest/Mitigated/requirements.txt):

```bash
# Create and activate a virtual environment
python -m venv .venv
# Windows:
.venv\Scripts\Activate.ps1
# Linux/macOS:
source .venv/bin/activate

# Install requirements
pip install -r Mitigated/requirements.txt
```

---

## Project Structure

```
CultureAwarenessTest/
├── Model/                      # Model implementations
│   ├── standard/               # Standard baselines (ResNet50V2, SVM, RFC)
│   ├── mitigated/              # Mitigated architectures with weight variance regularization
│   ├── diffusion/              # Denoising Diffusion Probabilistic Models (DDPM)
│   ├── discriminator/          # Classifiers to detect latent culture signatures
│   ├── adversarial/            # PGD adversarial attack generators
│   └── GeneralModel.py         # Base model wrapper class
├── Processing/
│   └── processing.py           # Core pipeline orchestration (ProcessingClass)
├── Utils/                      # Helper libraries
│   ├── Data/                   # Data load routines, culture splits, paths
│   ├── FileManager/            # Logging directories & outputs management
│   ├── Preprocessing/          # Standalone dataset prep (create_ds)
│   ├── Results/                # CM metrics & results aggregation
│   └── Visualizer/             # Confusion matrix & performance plotting
├── Mitigated/                  # Primary launch scripts & config environment files
│   ├── launch.py               # Main deep learning pipeline experiment loop
│   ├── launch_shallow.py       # Main shallow baseline pipeline loop
│   ├── environment.yml         # Conda environment definition
│   └── requirements.txt        # Pip dependencies list
├── LaunchFiles/                # Auxiliary execution and visualization scripts
│   ├── diffusion_step_plot.py  # DDPM step generator plotter
│   └── plot_standard_augmentation.py # Preview noise-based data augmentation
├── DatasetAnalysis/            # Feature analysis and clustering scripts
│   ├── examinate.py            # Deep feature embedding & distance examiner
│   └── k-means.py              # KMeans unsupervised clustering ($K=6$)
├── GradCam/                    # Model explainability suite
│   ├── launch.py               # Computes heatmaps for TP, TN, FP, FN categories
│   └── gradCam.py              # Core GradCAM activation map engine
├── OverallPipeline.ipynb       # Jupyter notebook aggregating final figures & tables
├── general_file.py             # Legacy monolithic single-file baseline compiler
└── LAUNCH_GUIDE.md             # Detailed execution reference guide
```

---

## Basic Usage

Detailed instructions for running each subsystem are located in [`LAUNCH_GUIDE.md`](file:///C:/Users/Utente/Desktop/CultureAwarenessTest/LAUNCH_GUIDE.md). Below are the primary quickstart commands:

### 1. Run Core Benchmarking Experiments
To run the standard control vs. bias-mitigated ResNet training loop:
```bash
python Mitigated/launch.py
```

### 2. Run Shallow Machine Learning Baselines
To evaluate SVM and Random Forest baselines:
```bash
python Mitigated/launch_shallow.py
```

### 3. Extract Dataset Features & Compute Cluster Distance Reports
To extract ResNet embeddings and output intra/inter-class distances to JSON:
```bash
python DatasetAnalysis/examinate.py
```

### 4. Generate GradCAM Interpretability Maps
To generate attention heatmaps across confusion matrix buckets (TP, TN, FP, FN):
```bash
python GradCam/launch.py
```

---

## ProcessingClass Configurations

To run custom pipelines programmatically, initialize and run `ProcessingClass` inside Python:

```python
from Processing.processing import ProcessingClass

# Initialize processor
procObj = ProcessingClass(
    shallow=0,            # 0 = Deep Learning (ResNet), 1 = Shallow (SVM/RFC)
    lamp=1,               # 1 = Lamps dataset, 0 = Carpets dataset
    gpu=True,             # Enable GPU acceleration
    memory_limit=6000,    # GPU virtual memory limit in MB
    basePath="./results/" # Log destination directory
)

# Run full pipeline
procObj.process(
    standard=0,           # 0 = Mitigated, 1 = Control
    culture=0,            # Culture index 0 as majority culture
    percent=0.05,         # Keep 5% minority data ratio
    augment=1,            # Enable classical augmentation
    diffusion=1,          # Enable diffusion augmentation
    only_minority_diffusion=1,
    parify_batches_diffusion=1
)

# Evaluate results
procObj.test(standard=0, culture=0)
```

---

## Troubleshooting

1. **GPU Out-Of-Memory (OOM)**:
   Ensure you allocate virtual memory limit parameters before training, as set up in [`Mitigated/launch.py`](file:///C:/Users/Utente/Desktop/CultureAwarenessTest/Mitigated/launch.py):
   ```python
   tf.config.experimental.set_virtual_device_configuration(
       gpus[0],
       [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=4000)] # Decrease to fit GPU
   )
   ```
2. **Missing Dataset Directories**:
   Verify raw input files are mapped to the directory targets defined inside [`Utils/Data/deep_paths.py`](file:///C:/Users/Utente/Desktop/CultureAwarenessTest/Utils/Data/deep_paths.py) and [`Utils/Data/shallow_paths.py`](file:///C:/Users/Utente/Desktop/CultureAwarenessTest/Utils/Data/shallow_paths.py).