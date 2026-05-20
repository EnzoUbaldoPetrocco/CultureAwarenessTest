import numpy as np
import matplotlib.pyplot as plt

# Set seed for reproducibility
np.random.seed(42)

def generate_imbalanced_plot():
    """
    Generates a scatter plot where Culture 2 is significantly 
    underrepresented compared to Culture 1.
    """
    
    # 1. Configuration for Publication Quality
    plt.rcParams.update({
        "font.family": "serif",
        "font.size": 11,
        "axes.titlesize": 12,
        "axes.labelsize": 11,
        "legend.fontsize": 9,
        "savefig.dpi": 300,
        "figure.autolayout": True
    })

    # 2. Data Generation
    # Culture 1: Majority (300 samples per class)
    # Culture 2: Underrepresented (25 samples per class)
    n_majority = 100
    n_minority = 5
    cov = [[0.12, 0], [0, 0.12]]

    # Culture 1 (Majority - Blue shades)
    c1_class0 = np.random.multivariate_normal([-0.75, 2.4], cov, n_majority)
    c1_class1 = np.random.multivariate_normal([-2.4, 0.75], cov, n_majority)

    # Culture 2 (Minority - Red/Orange shades)
    c2_class0 = np.random.multivariate_normal([2.4, -0.75], cov, n_minority)
    c2_class1 = np.random.multivariate_normal([0.75, -2.4], cov, n_minority)

    # 3. Plotting
    fig, ax = plt.subplots(figsize=(8, 6))

    # Plot Culture 1 (Majority)
    ax.scatter(c1_class0[:, 0], c1_class0[:, 1], 
               label='Culture 1: Class 0 ($n=100$)', alpha=0.4, 
               color='#004488', marker='o', s=20)
    ax.scatter(c1_class1[:, 0], c1_class1[:, 1], 
               label='Culture 1: Class 1 ($n=100$)', alpha=0.4, 
               color='#6699CC', marker='^', s=20)

    # Plot Culture 2 (Minority)
    # We increase 's' (size) slightly and alpha (opacity) to ensure visibility
    ax.scatter(c2_class0[:, 0], c2_class0[:, 1], 
               label='Culture 2: Class 0 ($n=5$)', alpha=0.9, 
               color='#994455', marker='s', s=40, edgecolors='black', linewidth=0.5)
    ax.scatter(c2_class1[:, 0], c2_class1[:, 1], 
               label='Culture 2: Class 1 ($n=5$)', alpha=0.9, 
               color='#EE99AA', marker='D', s=40, edgecolors='black', linewidth=0.5)

    # 4. Formal Styling
    ax.set_title('Distribution Imbalance: Majority vs. Underrepresented Cultures')
    ax.set_xlabel('Feature Dimension $X_1$')
    ax.set_ylabel('Feature Dimension $X_2$')
    
    ax.grid(True, linestyle='--', alpha=0.3)
    ax.legend(loc='upper right', frameon=True, facecolor='white', framealpha=1)

    # Visualizing the decision boundary or separation space if needed
    ax.axhline(0, color='black', linewidth=0.8, alpha=0.3)
    ax.axvline(0, color='black', linewidth=0.8, alpha=0.3)

    plt.show()

if __name__ == "__main__":
    generate_imbalanced_plot()