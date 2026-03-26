"""
Preprocessing Package - Data Augmentation Techniques

This package provides various data augmentation methods for increasing training data diversity.

Augmentation Techniques:
    - Classical Augmentation:
        * Random rotation
        * Gaussian noise addition
        * Random brightness adjustment
        * Random zoom/scale
    
    - Diffusion-Based Augmentation:
        * Uses diffusion models to generate realistic synthetic images
        * Targeted to minority cultures
    
    - Adversarial Augmentation:
        * Generates adversarial examples for robustness
        * PGD-based perturbations

Usage:
    prepObj = PreprocessingClass()
    X_augmented = prepObj.classical_augmentation(X, g=0.01)
"""
