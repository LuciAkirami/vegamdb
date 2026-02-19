import numpy as np
import myvector_db
import time


def calculate_recall(ground_truth, predicted, k):
    """
    Calculates the intersection between Ground Truth and Predicted results.
    Recall = (Number of Overlapping Indices) / K
    """
    total_recall = 0

    for gt, pred in zip(ground_truth, predicted):
        # Convert lists to sets to find intersection
        gt_set = set(gt)
        pred_set = set(pred)

        # Count how many correct neighbors we found
        intersection_count = len(gt_set.intersection(pred_set))

        # Add percentage for this specific query
        total_recall += intersection_count / k

    # Average over all queries
    return (total_recall / len(ground_truth)) * 100


def generate_clustered_data(n_vectors, dim, n_clusters_real=100):
    """
    Generates data that actually clumps together (like real-world semantic data).
    """
    print(f"Generating {n_vectors} CLUSTERED vectors...")

    # 1. Define 'Real' Cluster Centers (The Topics)
    # We pick 100 random points in space to be the "centers" of our topics.
    topic_centers = np.random.random((n_clusters_real, dim)).astype(np.float32)

    # 2. Generate Data around those centers
    data = []
    labels = []

    vectors_per_cluster = n_vectors // n_clusters_real

    for i in range(n_clusters_real):
        center = topic_centers[i]

        # Generate random noise (Gaussian)
        # scale=0.1 means the points are tight around the center.
        noise = np.random.normal(scale=0.1, size=(vectors_per_cluster, dim))

        # Add noise to the center
        cluster_points = center + noise
        data.append(cluster_points)

    # Flatten list of lists into one big array
    data = np.vstack(data).astype(np.float32)

    # Shuffle so they aren't ordered by cluster ID in memory
    np.random.shuffle(data)

    return data


def benchmark():
    # ---------------------------------------------------------
    # CONFIGURATION
    # ---------------------------------------------------------
    N_VECTORS = 100_000  # Size of Database
    DIM = 128  # Dimensions
    N_CLUSTERS = 100  # Number of centroids
    K = 10  # We want top 10 neighbors
    N_QUERIES = 100  # Number of queries to average

    # We will test these different probe levels
    # nprobe=1: Fastest, least accurate
    # nprobe=100: Slowest (Brute force equivalent), most accurate
    NPROBE_LIST = [1, 5, 10, 20, 50, 100]

    # ---------------------------------------------------------
    # 1. DATA GENERATION
    # ---------------------------------------------------------
    print(f"Generating {N_VECTORS} vectors (Dim: {DIM})...")

    # ---------------------------------------------------------
    # a. DATA GENERATION - Random
    # ---------------------------------------------------------
    # data = np.random.random((N_VECTORS, DIM)).astype(np.float32)
    # queries = np.random.random((N_QUERIES, DIM)).astype(np.float32)

    # ---------------------------------------------------------
    # b. DATA GENERATION - Pseudo Clustered
    # ---------------------------------------------------------
    # Instead of random noise, we generate 500 "topics"
    data = generate_clustered_data(N_VECTORS, DIM, n_clusters_real=500)

    # Queries should also come from the same distribution to mimic real users
    queries = generate_clustered_data(100, DIM, n_clusters_real=10)

    db = myvector_db.VegamDB()
    # db.set_dimension(128)
    print("Ingesting data into C++ DB...")
    for i in range(N_VECTORS):
        db.add_vector_numpy(data[i])

    # ---------------------------------------------------------
    # 2. ESTABLISH GROUND TRUTH
    # We run exact brute force search to know what the "Right Answers" are.
    # ---------------------------------------------------------
    print(f"Calculating Ground Truth for {N_QUERIES} queries using Flat Search...")
    start_flat = time.time()
    ground_truth_results = []

    for i in range(N_QUERIES):
        # search() is the brute force O(N) method
        res = db.search(queries[i], K)
        ground_truth_results.append(res.ids)

    time_flat = time.time() - start_flat
    avg_flat_ms = (time_flat / N_QUERIES) * 1000
    print(f"Ground Truth Calculation Complete.")
    print(f"Avg Flat Search Latency: {avg_flat_ms:.4f} ms")

    # ---------------------------------------------------------
    # 3. BUILD INDEX
    # ---------------------------------------------------------
    print(f"\nTraining IVF Index ({N_CLUSTERS} clusters)...")
    start_train = time.time()
    db.use_ivf_index(n_clusters=100, max_iters=20)
    db.build_index()
    print(f"Training Time: {time.time() - start_train:.4f}s")

    # ---------------------------------------------------------
    # 4. RUN PROBE SWEEP
    # ---------------------------------------------------------
    print("\n--- RESULTS TABLE ---")
    header = f"{'NPROBE':<10} | {'AVG LATENCY (ms)':<18} | {'SPEEDUP':<10} | {'ACCURACY (%)':<12}"
    print("-" * len(header))
    print(header)
    print("-" * len(header))

    for nprobe in NPROBE_LIST:
        start_probe = time.time()
        ivf_results = []

        # Run all queries with this nprobe setting
        for i in range(N_QUERIES):
            params = myvector_db.IVFSearchParams()
            params.n_probe = nprobe
            res = db.search(queries[i], K, params)
            ivf_results.append(res.ids)

        # Metrics
        total_time = time.time() - start_probe
        avg_time_ms = (total_time / N_QUERIES) * 1000
        speedup = time_flat / total_time
        accuracy = calculate_recall(ground_truth_results, ivf_results, K)

        # Print Row
        print(
            f"{nprobe:<10} | {avg_time_ms:<18.4f} | {speedup:<10.1f} | {accuracy:<12.1f}"
        )

    print("-" * len(header))


if __name__ == "__main__":
    benchmark()
