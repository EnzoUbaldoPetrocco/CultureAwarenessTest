import numpy as np
import matplotlib.pyplot as plt

# --- Set Global Academic Style (Compatible version) ---
plt.rcParams.update({
    "font.family": "serif",
    "font.size": 9,
    "axes.titlesize": 10,
    "savefig.dpi": 300
})

np.random.seed(42)

# --- Parameters ---
N_MAJ, N_MIN = 250, 25
EPSILON = 0.4  # PGD Constraint Radius

def get_base_data():
    cov = [[0.08, 0], [0, 0.08]]
    # Culture 1 (Majority) - Left/Top
    c1_0 = np.random.multivariate_normal([-0.75, 2.4], cov, N_MAJ)
    c1_1 = np.random.multivariate_normal([-0.7, 0.7], cov, N_MAJ)
    # Culture 2 (Minority) - Right/Bottom
    c2_0 = np.random.multivariate_normal([2.4, -0.75], cov, N_MIN)
    c2_1 = np.random.multivariate_normal([0.7, -0.8], cov, N_MIN)
    return c1_0, c1_1, c2_0, c2_1

c1_0, c1_1, c2_0, c2_1 = get_base_data()

# Define formal colors
c_maj0 = '#004488' # Dark Blue
c_maj1 = '#6699CC' # Light Blue
c_min0 = '#994455' # Dark Red
c_min1 = '#EE99AA' # Pink

# Create 2x3 grid
fig, axes = plt.subplots(2, 3, figsize=(14, 9))
axes = axes.flatten()

titles = [
    "Imbalanced Baseline",
    "Random Transformation (RT)",
    "Adversarial PGD (Global)",
    "Adversarial PGD (Class-wise)",
    "Diffusion (RT + Minority)",
    "Diffusion (RT + Both)"
]

for i, ax in enumerate(axes):
    ax.set_title(titles[i], fontweight='bold')
    ax.set_xlim(-2.5, 4); ax.set_ylim(-2.5, 4)
    ax.grid(True, linestyle=':', alpha=0.5)
    
    # Plot Original Data Background (Visible in all plots)
    # Using alpha to make them sit in the background for Figs 2-6
    alpha_base_maj = 0.6 if i == 0 else 0.15
    alpha_base_min = 0.9 if i == 0 else 0.4
    
    ax.scatter(c1_0[:,0], c1_0[:,1], c=c_maj0, s=15, alpha=alpha_base_maj, label='C1-C0 (Maj)' if i==0 else "")
    ax.scatter(c1_1[:,0], c1_1[:,1], c=c_maj1, s=15, alpha=alpha_base_maj, label='C1-C1 (Maj)' if i==0 else "")
    ax.scatter(c2_0[:,0], c2_0[:,1], c=c_min0, s=35, alpha=alpha_base_min, edgecolors='k', lw=0.5, label='C2-C0 (Min)' if i==0 else "")
    ax.scatter(c2_1[:,0], c2_1[:,1], c=c_min1, s=35, alpha=alpha_base_min, edgecolors='k', lw=0.5, label='C2-C1 (Min)' if i==0 else "")

    if i == 1: # RT applied to BOTH minority classes
        j0 = np.random.normal(0, 0.15, c2_0.shape)
        j1 = np.random.normal(0, 0.15, c2_1.shape)
        ax.scatter((c2_0 + j0)[:,0], (c2_0 + j0)[:,1], c=c_min0, marker='+', s=30, alpha=0.6, label='RT C0')
        ax.scatter((c2_1 + j1)[:,0], (c2_1 + j1)[:,1], c=c_min1, marker='+', s=30, alpha=0.6, label='RT C1')

    elif i == 2: # PGD Global
        # Compute global centroids for Culture 1 and Culture 2
        mu_c1 = np.mean(np.vstack((c1_0, c1_1)), axis=0)
        mu_c2 = np.mean(np.vstack((c2_0, c2_1)), axis=0)
        
        # Single global direction
        dir_global = (mu_c2 - mu_c1)
        dir_global /= np.linalg.norm(dir_global)
        
        # Attack moves BOTH classes in the exact same global direction
        pgd_c0 = c1_0 + dir_global * EPSILON
        pgd_c1 = c1_1 + dir_global * EPSILON
        
        ax.scatter(pgd_c0[:,0], pgd_c0[:,1], c=c_maj0, marker='.', s=20, alpha=0.7)
        ax.scatter(pgd_c1[:,0], pgd_c1[:,1], c=c_maj1, marker='.', s=20, alpha=0.7)
        
        # Plot Global Decision Boundary (Orthogonal to shift)
        midpoint = (mu_c1 + mu_c2) / 2
        ortho_dir = np.array([-dir_global[1], dir_global[0]])
        ax.axline(midpoint, midpoint + ortho_dir, color='black', lw=1.5, ls='-', alpha=0.7, label='Global Boundary')
        
        # Visual constraint marker
        ax.add_patch(plt.Circle(c1_0[0], EPSILON, color='black', fill=False, lw=0.8, ls='--'))

    elif i == 3: # PGD Class-wise
        # Compute specific directions for each class
        dir_0 = np.mean(c2_0, axis=0) - np.mean(c1_0, axis=0)
        dir_0 /= np.linalg.norm(dir_0)
        
        dir_1 = np.mean(c2_1, axis=0) - np.mean(c1_1, axis=0)
        dir_1 /= np.linalg.norm(dir_1)
        
        # Attack moves each class along its own semantic path
        pgd_c0 = c1_0 + dir_0 * EPSILON
        pgd_c1 = c1_1 + dir_1 * EPSILON
        
        ax.scatter(pgd_c0[:,0], pgd_c0[:,1], c=c_maj0, marker='.', s=20, alpha=0.7)
        ax.scatter(pgd_c1[:,0], pgd_c1[:,1], c=c_maj1, marker='.', s=20, alpha=0.7)
        
        # Plot Conditional Boundaries
        mid_0 = (np.mean(c1_0, axis=0) + np.mean(c2_0, axis=0)) / 2
        ax.axline(mid_0, mid_0 + np.array([-dir_0[1], dir_0[0]]), color='black', lw=1, ls='--', alpha=0.6, label='C0 Boundary')
        
        mid_1 = (np.mean(c1_1, axis=0) + np.mean(c2_1, axis=0)) / 2
        ax.axline(mid_1, mid_1 + np.array([-dir_1[1], dir_1[0]]), color='gray', lw=1, ls='--', alpha=0.6, label='C1 Boundary')

    elif i == 4: # Diffusion Minority
        # Structured synthesis for both minority classes
        synth_0 = np.random.multivariate_normal([2.4, -0.75], [[0.1, 0.03], [0.03, 0.1]], 120)
        synth_1 = np.random.multivariate_normal([0.7, -0.8], [[0.1, 0.03], [0.03, 0.1]], 120)
        ax.scatter(synth_0[:,0], synth_0[:,1], c=c_min0, marker='s', s=10, alpha=0.3)
        ax.scatter(synth_1[:,0], synth_1[:,1], c=c_min1, marker='s', s=10, alpha=0.3)

    elif i == 5: # Diffusion Both
        # Broader synthesis over the entire distribution space
        cov_broad = [[0.15, 0], [0, 0.15]]
        s_maj0 = np.random.multivariate_normal([-0.75, 2.4], cov_broad, 200)
        s_maj1 = np.random.multivariate_normal([-0.7, 0.7], cov_broad, 200)
        s_min0 = np.random.multivariate_normal([2.4, -0.75], cov_broad, 150)
        s_min1 = np.random.multivariate_normal([0.7, -0.8], cov_broad, 150)
        
        ax.scatter(s_maj0[:,0], s_maj0[:,1], c=c_maj0, marker='s', s=6, alpha=0.15)
        ax.scatter(s_maj1[:,0], s_maj1[:,1], c=c_maj1, marker='s', s=6, alpha=0.15)
        ax.scatter(s_min0[:,0], s_min0[:,1], c=c_min0, marker='s', s=6, alpha=0.15)
        ax.scatter(s_min1[:,0], s_min1[:,1], c=c_min1, marker='s', s=6, alpha=0.15)

    # Add subplot-specific legends for the boundaries/features if they exist
    if i in [1, 2, 3]:
        ax.legend(loc='upper right', fontsize=8)

# Main Legend positioning (grabbing baseline labels)
handles, labels = axes[0].get_legend_handles_labels()
fig.legend(handles, labels, loc='lower center', ncol=4, bbox_to_anchor=(0.5, 0.02))

# Use tight_layout to prevent overlap
plt.tight_layout(rect=[0, 0.06, 1, 1]) 
plt.savefig('mock_strategies_comparison.pdf')  # Save as PDF for high-quality vector graphics
plt.show()