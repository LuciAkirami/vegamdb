import numpy as np
import myvector_db
import matplotlib.pyplot as plt  # Optional: Only if you want to see the plot


def generate_blobs(n_points, dim=2):
    # Create 3 distinct clusters
    data = []

    # Blob 1: Centered at (10, 10)
    data.append(np.random.randn(n_points, dim) + 10)

    # Blob 2: Centered at (20, 20)
    data.append(np.random.randn(n_points, dim) + 20)

    # Blob 3: Centered at (30, 10)
    data.append(np.random.randn(n_points, dim) + [30, 10])

    return np.vstack(data).astype(np.float32)


# 1. Generate Data
print("Generating 3 blobs of data...")
data = generate_blobs(100, dim=2)  # 300 points total

# 2. Train K-Means
print("Training K-Means (K=3)...")
# Args: K=3, MaxIters=10, Dim=2
kmeans = myvector_db.KMeans(3, 10, 2)

# Note: We must explicitly cast to list because we haven't optimized
# the KMeans input for NumPy pointers yet! (Slow but fine for testing)
index = kmeans.train(data.tolist())

# 3. Print Results
print("\nFound Centroids:")
for i, c in enumerate(index.centroids):
    print(f"Cluster {i}: {c} (Count: {len(index.buckets[i])})")

# 4. Optional: Plot (if you have matplotlib installed)
try:
    colors = ["r", "g", "b"]
    for i in range(3):
        bucket_indices = index.buckets[i]
        points = data[bucket_indices]
        plt.scatter(points[:, 0], points[:, 1], c=colors[i], label=f"Cluster {i}")

        # Plot Centroid
        cx, cy = index.centroids[i]
        plt.scatter(cx, cy, c="black", marker="x", s=200)

    plt.legend()
    plt.title("C++ K-Means Result")
    plt.show()
except ImportError:
    print("\n(Install matplotlib to see the graph: pip install matplotlib)")
