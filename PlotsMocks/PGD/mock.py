import numpy as np
import matplotlib.pyplot as plt
from sklearn.neighbors import KDTree
from sklearn.datasets import make_moons, make_circles

def generate_custom_distribution(shape_type='moons', n_samples=500, center=(0, 0), noise=0.1):
    """
    Generates non-convex 2D distributions.
    
    Parameters:
    - shape_type: 'moons', 'circles', or 'spiral'
    - n_samples: Total number of points
    - center: (x, y) offset to move the distribution
    - noise: Standard deviation of Gaussian noise added to the data
    """
    if shape_type == 'moons':
        X, _ = make_moons(n_samples=n_samples, noise=noise, random_state=42)
    
    elif shape_type == 'circles':
        X, _ = make_circles(n_samples=n_samples, noise=noise, factor=0.5, random_state=42)
    
    elif shape_type == 'spiral':
        # Manually generating a non-convex spiral
        n = np.sqrt(np.random.rand(n_samples, 1)) * 780 * (2 * np.pi) / 360
        d1x = -np.cos(n) * n + np.random.randn(n_samples, 1) * noise * 10
        d1y = np.sin(n) * n + np.random.randn(n_samples, 1) * noise * 10
        X = np.hstack((d1x, d1y)) / 10  # Scale down
        
    else:
        raise ValueError("Choose shape_type from 'moons', 'circles', or 'spiral'")

    # Apply the center offset
    X += np.array(center)
    
    return X

def pgd_to_manifold(x_start, target_distribution, steps=15, alpha=0.05, g=0.6):
    """
    PGD that pushes a sample toward a complex target distribution (manifold).
    """
    # We use a KDTree to simulate the Discriminator's gradient
    # The gradient points toward the nearest neighbor in the target set
    tree = KDTree(target_distribution)
    
    x_iter = x_start.copy()
    history = [x_start.copy()]
    
    for _ in range(steps):
        # 1. Simulate Gradient: Find the closest point in the target distribution
        dist, ind = tree.query(x_iter.reshape(1, -1), k=1)
        closest_point = target_distribution[ind[0][0]]
        
        # Direction toward the manifold
        direction = closest_point - x_iter
        norm = np.linalg.norm(direction)
        
        if norm < 1e-5: break # Already there
            
        grad = direction / norm
        
        # 2. Update Step
        x_iter = x_iter + alpha * grad
        
        # 3. Projection Step: Ensure we haven't moved further than 'g' from origin
        total_displacement = x_iter - x_start
        total_dist = np.linalg.norm(total_displacement)
        if total_dist > g:
            x_iter = x_start + (total_displacement / total_dist) * g
            
        history.append(x_iter.copy())
        
    return np.array(history)

# --- Scenario Setup ---
# Dense Distribution (Source) - A tight cluster
source_data = np.random.normal(loc=[1.5, 0.8], scale=0.1, size=(1, 2)) 

# Less Dense Distribution (Target) - The Moon shape
target_manifold = generate_custom_distribution(shape_type='moons', n_samples=10, center=(0, 0), noise=0.05)

# Run PGD for a single sample
sample = source_data[0]
g = 0.25
path = pgd_to_manifold(sample, target_manifold, g=g, alpha=0.04, steps=5)

# --- Visualization ---
plt.figure(figsize=(10, 6))
plt.scatter(source_data[:, 0], source_data[:, 1], s=5, color='blue', alpha=0.3, label='Source Culture (Dense Sample)')
plt.scatter(target_manifold[:, 0], target_manifold[:, 1], s=5, color='gray', alpha=0.3, label='Target Culture (Manifold)')
plt.plot(path[:, 0], path[:, 1], 'o-', color='orangered', markersize=4, label='PGD Path')
plt.scatter(sample[0], sample[1], color='blue', s=100, edgecolors='black', label='Original Dense Sample', zorder=5)
plt.scatter(path[-1, 0], path[-1, 1], color='green', s=100, edgecolors='black', label='Pushed Sample', zorder=5)

# Visualize the Constraint 'g'
circle = plt.Circle(sample, g, color='blue', fill=False, linestyle='--', alpha=0.3, label='Constraint Budget (g)')
plt.gca().add_patch(circle)

plt.title("PGD Pushing a Sample toward a Non-Convex Target")
plt.legend()
plt.axis('equal')
plt.show()