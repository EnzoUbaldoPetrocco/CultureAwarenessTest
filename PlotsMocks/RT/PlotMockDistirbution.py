import numpy as np
import matplotlib.pyplot as plt
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

def apply_random_displacement(data, g):
    """
    Moves each point in a random direction by a distance d <= g.
    
    Parameters:
    - data: NumPy array of shape (n_samples, 2)
    - g: Maximum displacement distance
    """
    n_samples = data.shape[0]
    
    # 1. Generate random angles (0 to 2*pi)
    angles = np.random.uniform(0, 2 * np.pi, n_samples)
    
    # 2. Generate random magnitudes (d <= g)
    # Using sqrt for uniform distribution within the area of the disk
    magnitudes = g * np.sqrt(np.random.uniform(0, 1, n_samples))
    
    # 3. Convert polar coordinates to Cartesian (dx, dy)
    dx = magnitudes * np.cos(angles)
    dy = magnitudes * np.sin(angles)
    
    # 4. Apply the displacement
    displacements = np.vstack((dx, dy)).T
    transformed_data = data + displacements
    
    return transformed_data



# 1. Generate the base data
# Lowering n_samples to 15 makes the displacement lines much clearer
n_samples = 15
original_data = generate_custom_distribution(shape_type='moons', n_samples=n_samples, center=(5, 5), noise=0.02)

# 2. Apply displacement to the SAME data
g_val = 0.1
transformed_data = apply_random_displacement(original_data, g=g_val)

plt.figure(figsize=(10, 8))

# Plot the "Ghost" points (Original position)
plt.scatter(original_data[:, 0], original_data[:, 1], color='gray', alpha=0.3, label='Original Position', s=50)

# Plot the "New" points (Displaced position)
plt.scatter(transformed_data[:, 0], transformed_data[:, 1], color='teal', label='Displaced Position', s=50)

# 3. Draw the Displacement Vectors
for i in range(n_samples):
    plt.annotate("", 
                 xy=(transformed_data[i, 0], transformed_data[i, 1]), 
                 xytext=(original_data[i, 0], original_data[i, 1]),
                 arrowprops=dict(arrowstyle="->", color="orange", lw=1.5, alpha=0.8))

plt.title(f"Random Displacement Visualization (max distance g = {g_val})", fontsize=14)
plt.xlabel("X coordinate")
plt.ylabel("Y coordinate")
plt.legend()
plt.axis('equal')
plt.grid(True, linestyle='--', alpha=0.4)

plt.show()
