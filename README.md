# CultureAwarenessTest

## Overview

CultureAwarenessTest is a machine learning research project designed to investigate and mitigate cultural bias in deep learning models for image classification. The project focuses on addressing the hypothesis that trained models may exhibit systematic bias toward specific cultures when classifying household objects (lamps and carpets) from different cultural origins.

The project implements various strategies to identify, measure, and reduce cultural bias, including custom regularization techniques, diffusion-based synthetic data generation, and adversarial robustness testing.

## Key Features

- **Bias Detection**: Uses discriminator models to detect cultural bias in predictions
- **Mitigation Strategies**: Implements custom regularization to reduce culture-specific weight variances
- **Synthetic Data Generation**: Employs denoising diffusion probabilistic models for minority culture data augmentation
- **Adversarial Testing**: Evaluates model robustness against culture-based adversarial attacks
- **Comprehensive Evaluation**: Provides detailed metrics, confusion matrices, and GradCam visualizations

## Installation

### Prerequisites

- Python 3.8+
- TensorFlow 2.10.1
- CUDA-compatible GPU (recommended for training)

### Dependencies

Install the required packages using pip:

```bash
pip install -r requirements.txt
```

For the mitigated models specifically:

```bash
pip install -r Mitigated/requirements.txt
```

Key dependencies include:
- TensorFlow 2.10.1
- Keras 2.10.0
- scikit-learn 1.6.1
- OpenCV
- NumPy, Pandas, Matplotlib

## Project Structure

```
CultureAwarenessTest/
├── Model/                      # ML model implementations
│   ├── standard/               # Baseline models (SVM, RFC, ResNet)
│   ├── mitigated/              # Bias-mitigated models with regularization
│   ├── diffusion/              # Diffusion models for data generation
│   ├── discriminator/          # Culture discrimination classifiers
│   ├── adversarial/            # Adversarial attack generators
│   └── GeneralModel.py         # Base model class
├── Processing/
│   └── processing.py           # Main pipeline orchestration
├── Utils/
│   ├── Data/                   # Data loading utilities
│   ├── FileManager/            # Result logging
│   ├── Preprocessing/          # Data augmentation
│   ├── Results/                # Metrics computation
│   └── Visualizer/             # Plotting and visualization
├── LaunchFiles/                # Execution scripts
├── Mitigated/                  # Main execution and results
├── DataAnalysis/               # Analysis scripts and results
├── GradCam/                    # Model interpretability
├── OTHERRESULTS/               # Alternative experiment results
├── OverallPipeline.ipynb       # Comprehensive analysis notebook
├── requirements.txt            # Project dependencies
└── LICENSE                     # License information
```

## Usage

### Basic Execution

Run the main mitigated model training:

```bash
python Mitigated/launch.py
```

### Custom Configuration

The main entry point is `Mitigated/launch.py`. Key parameters include:

- `percent`: Percentage of minority culture data (e.g., 0.05 = 5%)
- `basePath`: Output directory path
- `todo_configs`: List of tuples defining experiment configurations

Each configuration tuple follows: `(standard, lamp, culture, diffusion, only_min, parify, augment)`

Where:
- `standard`: 0=Mitigated model, 1=Standard model
- `lamp`: 0=Carpets dataset, 1=Lamps dataset
- `culture`: Majority culture ID (0-2)
- `diffusion`: Enable diffusion-based augmentation
- `only_min`: Generate data only for minority cultures
- `parify`: Enable culture-balanced batch sampling
- `augment`: Enable classical augmentation

### Using ProcessingClass

```python
from Processing.processing import ProcessingClass

# Initialize processor
procObj = ProcessingClass(shallow=0, lamp=1, gpu=True, memory_limit=13000)

# Run full pipeline
procObj.process(
    standard=0,           # Use mitigated model
    culture=0,            # Culture 0 as majority
    percent=0.05,         # 5% minority data
    augment=1,            # Enable augmentation
    diffusion=1,          # Enable diffusion
    only_minority_diffusion=1,
    parify_batches_diffusion=1
)

# Test the model
procObj.test(standard=0, culture=0)
```

### Custom Model Training

```python
from Model.mitigated.mitigated_models import MitigatedModels

# Initialize mitigated model
model = MitigatedModels(
    type="DL",
    culture=0,
    n_cultures=3,
    lambda_index=15  # Regularization strength
)

# Train the model
model.fit(train_data, val_data, epochs=15, batch_size=32)
```

## Data Flow

1. **Data Loading**: Images are loaded per culture and label using DataClass
2. **Preprocessing**: Train/validation/test splits with culture stratification
3. **Augmentation** (optional):
   - Classical: Rotation, noise, brightness adjustments
   - Diffusion: Synthetic image generation for minority cultures
4. **Training**: Models trained with optional culture balancing
5. **Evaluation**: Testing with confusion matrices and per-culture metrics

## Models and Algorithms

### Standard Models
- **SVM**: Support Vector Machines with linear/RBF kernels
- **Random Forest**: Ensemble classification
- **ResNet50V2**: Pre-trained convolutional neural network

### Mitigated Models
- **Custom Regularization**: Minimizes variance of culture-specific weight variances
- **Culture-Inclusive Loss**: Measures per-culture loss variance
- **Architecture**: ResNet50V2 with modified final layers

### Diffusion Models
- **Denoising Diffusion**: Generates synthetic images for data balancing
- **Targeted Generation**: Focuses on minority cultures

### Discriminator Models
- **Culture Classification**: SVM-based binary classifiers for bias detection

### Adversarial Models
- **PGD Attacks**: Projected Gradient Descent for adversarial examples
- **Culture-Specific Perturbations**: Tests robustness across cultures

## Analysis and Visualization

### Overall Analysis
Run the comprehensive analysis notebook:

```bash
jupyter notebook OverallPipeline.ipynb
```

### Visualization Tools
- **Confusion Matrices**: Per-culture classification performance
- **GradCam**: Model interpretability visualizations
- **Result Plotting**: Custom plotting utilities in `Utils/Visualizer/`

## Results Structure

Results are organized by culture and confidence thresholds:
- `CI/`, `CJ/`, `CS/`, `LC/`, `LF/`, `LT/`: Culture-specific folders
- Subfolders for confidence levels: `0.05/`, `0.2/`, `0.3/`
- Contains generated images, discriminator outputs, and preprocessing logs

## Research Questions

1. **Bias Detection**: Does cultural bias exist in standard classification models?
2. **Mitigation Effectiveness**: Can custom regularization reduce cultural disparities?
3. **Data Augmentation**: How effective is diffusion-based synthetic data generation?
4. **Robustness**: Are models vulnerable to culture-based adversarial attacks?

## Troubleshooting

### Common Issues

1. **Memory Errors**: Reduce `memory_limit` in ProcessingClass or use CPU mode
2. **CUDA Errors**: Ensure compatible TensorFlow-GPU version and CUDA installation
3. **Data Loading**: Verify dataset paths and culture label mappings
4. **Training Failures**: Check model configurations and batch sizes

### Performance Tips

- Use GPU acceleration for faster training
- Start with smaller datasets for initial testing
- Monitor memory usage with `memory_limit` parameter
- Use `shallow=1` for quick prototyping

## License

This project is licensed under the terms specified in the LICENSE file.

## Documentation

- [API Documentation](docs/API.md) - Detailed class and method documentation
- [Contributing Guide](CONTRIBUTING.md) - Development and contribution guidelines
- [OverallPipeline.ipynb](OverallPipeline.ipynb) - Comprehensive analysis notebook

## Contributing

For contributions or questions about the research methodology, please refer to the [contributing guide](CONTRIBUTING.md) and code documentation.