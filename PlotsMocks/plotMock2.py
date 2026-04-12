import numpy as np
import matplotlib.pyplot as plt

# Set seed for reproducibility
np.random.seed(42)

def generate_paper_plot():
    """
    Generates a high-quality scatter plot representing two cultures,
    each containing two distinct classes.
    """
    
    # 1. Configuration for Publication Quality
    plt.rcParams.update({
        "font.family": "serif",
        "font.size": 11,
        "axes.titlesize": 12,
        "axes.labelsize": 11,
        "legend.fontsize": 10,
        "savefig.dpi": 300,  # High resolution for papers
        "figure.autolayout": True
    })

    # 2. Data Generation
    # Culture 1: Located in the top-left quadrant
    # Culture 2: Located in the bottom-right quadrant
    n_samples = 150
    cov = [[0.2, 0], [0, 0.2]]  # Tight variance for clarity

    # Culture 1 (Blue tones)
    c1_class0 = np.random.multivariate_normal([-0.75, 2.4], cov, n_samples)
    c1_class1 = np.random.multivariate_normal([-2.4, 0.75], cov, n_samples)

    # Culture 2 (Red/Orange tones)
    c2_class0 = np.random.multivariate_normal([2.4, -0.75], cov, n_samples)
    c2_class1 = np.random.multivariate_normal([0.75, -2.4], cov, n_samples)

    # 3. Plotting
    fig, ax = plt.subplots(figsize=(8, 6))

    # Plot Culture 1
    ax.scatter(c1_class0[:, 0], c1_class0[:, 1], 
               label='Culture 1: Class 0', alpha=0.6, color='#004488', marker='o', s=25)
    ax.scatter(c1_class1[:, 0], c1_class1[:, 1], 
               label='Culture 1: Class 1', alpha=0.6, color='#6699CC', marker='^', s=25)

    # Plot Culture 2
    ax.scatter(c2_class0[:, 0], c2_class0[:, 1], 
               label='Culture 2: Class 0', alpha=0.6, color='#994455', marker='s', s=25)
    ax.scatter(c2_class1[:, 0], c2_class1[:, 1], 
               label='Culture 2: Class 1', alpha=0.6, color='#EE99AA', marker='D', s=25)

    # 4. Formal Styling
    ax.set_title('2-D Representation of Two Cultures with Distinct Classes', fontsize=12, fontweight='bold')
    ax.set_xlabel('Feature Dimension $X_1$')
    ax.set_ylabel('Feature Dimension $X_2$')
    
    # Add a subtle grid
    ax.grid(True, linestyle='--', alpha=0.4)
    
    # Place legend outside or in an empty area
    ax.legend(loc='best', frameon=True, shadow=False, borderpad=1)

    # Optional: Save as PDF/EPS for vector graphics support in LaTeX
    # plt.savefig('distribution_plot.pdf')
    
    plt.show()

if __name__ == "__main__":
    generate_paper_plot()