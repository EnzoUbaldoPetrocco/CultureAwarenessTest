import numpy as np
import matplotlib.pyplot as plt

def generate_paper_figures():
    np.random.seed(42)
    
    # Define centers for (Class, Culture)
    # C0: Left (-2), C1: Right (2) | K0: Bottom (-2), K1: Top (2)
    centers = {
        (0, 0): [-2, -2], (1, 0): [2, -2], # Culture 0 (Well represented)
        (0, 1): [-2, 2],  (1, 1): [2, 2]   # Culture 1 (Underrepresented)
    }
    
    # Original Data
    c0_k0 = np.random.normal(centers[(0,0)], 0.3, (30, 2))
    c1_k0 = np.random.normal(centers[(1,0)], 0.3, (30, 2))
    c0_k1 = np.random.normal(centers[(0,1)], 0.3, (4, 2))  # Sparse
    c1_k1 = np.random.normal(centers[(1,1)], 0.3, (4, 2))  # Sparse

    # --- FIGURE 1: RANDOM TRANSFORMATIONS ---
    plt.figure(figsize=(8, 8))
    plt.axhline(0, color='black', alpha=0.3); plt.axvline(0, color='black', alpha=0.3)
    
    # Plot originals
    plt.scatter(c0_k0[:,0], c0_k0[:,1], c='blue', label='C0, K0', alpha=0.6)
    plt.scatter(c1_k0[:,0], c1_k0[:,1], c='red', label='C1, K0', alpha=0.6)
    plt.scatter(c0_k1[:,0], c0_k1[:,1], c='blue', marker='x', s=100, label='C0, K1 (Sparse)')
    plt.scatter(c1_k1[:,0], c1_k1[:,1], c='red', marker='x', s=100, label='C1, K1 (Sparse)')
    
    # Random Trans: Just jittering the sparse points
    rt_c0 = c0_k1 + np.random.normal(0, 0.4, (4, 2))
    rt_c1 = c1_k1 + np.random.normal(0, 0.4, (4, 2))
    plt.scatter(rt_c0[:,0], rt_c0[:,1], c='blue', alpha=0.2, s=20)
    plt.scatter(rt_c1[:,0], rt_c1[:,1], c='red', alpha=0.2, s=20)
    
    plt.title("Figure 1: Random Transformations (Local Only)")
    plt.legend(); plt.xlim(-4, 4); plt.ylim(-4, 4)
    plt.show()

    
generate_paper_figures()


import numpy as np
import matplotlib.pyplot as plt

def plot_pgm_with_discriminator():
    np.random.seed(42)
    
    # 1. Setup the "Cultural Field" (The Discriminator)
    # Let's assume K=1 is a region in the top-left and K=0 is bottom-right
    x_range = np.linspace(-4, 4, 100)
    y_range = np.linspace(-4, 4, 100)
    X, Y = np.meshgrid(x_range, y_range)
    
    # Mock Discriminator: Sigmoid of a linear boundary (x + y)
    # K=1 (Target) is yellow/high, K=0 (Source) is purple/low
    Z = 1 / (1 + np.exp(-(X + Y - 1))) 

    # 2. Generate Initial Data (Majority Culture K=0)
    c0_k0 = np.random.multivariate_normal([-1.5, -1.5], [[0.4, 0], [0, 0.4]], 12)
    c1_k0 = np.random.multivariate_normal([1.5, -1.5], [[0.4, 0], [0, 0.4]], 12)

    plt.figure(figsize=(10, 8))
    
    # Plot the Discriminator Function as a background
    contour = plt.contourf(X, Y, Z, levels=20, cmap='RdYlGn', alpha=0.3)
    cbar = plt.colorbar(contour)
    cbar.set_label('Discriminator Output $D(x)$ (Target Culture Probability)', rotation=270, labelpad=15)

    def run_pgm_and_plot(points, color, label):
        steps = 7
        eta = 0.5 # Step size
        epsilon = 2.0 # Constraint
        
        for i, p_start in enumerate(points):
            path = [p_start]
            current = p_start.copy()
            
            for s in range(steps):
                # The gradient of our mock discriminator (X+Y-1) is [1, 1]
                grad = np.array([1.0, 1.2]) + np.random.normal(0, 0.1, 2)
                next_step = current + eta * grad
                
                # Projection check
                if np.linalg.norm(next_step - p_start) > epsilon:
                    next_step = p_start + (next_step - p_start) / np.linalg.norm(next_step - p_start) * epsilon
                
                path.append(next_step)
                current = next_step
            
            path = np.array(path)
            # Plot path
            plt.plot(path[:, 0], path[:, 1], color=color, alpha=0.4, linestyle='--', linewidth=1)
            # Plot start and end with larger dots
            if i == 0:
                plt.scatter(path[0, 0], path[0, 1], c=color, s=150, edgecolors='k', marker='o', label=f'Original {label}')
                plt.scatter(path[-1, 0], path[-1, 1], c=color, s=200, edgecolors='white', linewidth=2, marker='*', label=f'PGM {label}')
            else:
                plt.scatter(path[0, 0], path[0, 1], c=color, s=150, edgecolors='k')
                plt.scatter(path[-1, 0], path[-1, 1], c=color, s=200, edgecolors='white', linewidth=2, marker='*')
            
            # Draw an arrow at the final step
            plt.arrow(path[-2,0], path[-2,1], (path[-1,0]-path[-2,0])*0.8, (path[-1,1]-path[-2,1])*0.8, 
                      head_width=0.03, color=color, alpha=0.8)

    # Run for both classes
    run_pgm_and_plot(c0_k0, 'blue', 'Class 0')
    run_pgm_and_plot(c1_k0, 'red', 'Class 1')

    plt.title("Adversarial PGM: Pushing Samples across the Cultural Discriminator Field", fontsize=14)
    plt.xlabel("Feature Space $x_1$")
    plt.ylabel("Feature Space $x_2$")
    plt.legend(loc='lower right')
    plt.grid(alpha=0.2)
    plt.tight_layout()
    plt.show()

plot_pgm_with_discriminator()