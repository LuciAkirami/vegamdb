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
        gt_set = set(gt)
        pred_set = set(pred)
        intersection_count = len(gt_set.intersection(pred_set))
        total_recall += intersection_count / k

    return (total_recall / len(ground_truth)) * 100


def generate_clustered_data(n_vectors, dim, n_clusters_real=100):
    print(f"Generating {n_vectors} CLUSTERED vectors...")
    topic_centers = np.random.random((n_clusters_real, dim)).astype(np.float32)
    data = []
    vectors_per_cluster = n_vectors // n_clusters_real

    for i in range(n_clusters_real):
        center = topic_centers[i]
        noise = np.random.normal(scale=0.1, size=(vectors_per_cluster, dim))
        cluster_points = center + noise
        data.append(cluster_points)

    data = np.vstack(data).astype(np.float32)
    np.random.shuffle(data)
    return data


def benchmark():
    # ---------------------------------------------------------
    # CONFIGURATION
    # ---------------------------------------------------------
    N_VECTORS = 100000
    DIM = 128
    K = 10
    N_QUERIES = 100

    # Annoy Param: Max items in a leaf before splitting
    K_LEAF = 50

    # We will test different Forest Sizes (Number of Trees)
    # More Trees = Better Chance of finding neighbors = Slower Search
    TREE_COUNTS = [1, 2, 5, 10, 50, 100, 200, 500]

    # ---------------------------------------------------------
    # 1. DATA GENERATION
    # ---------------------------------------------------------
    print(f"Generating {N_VECTORS} vectors (Dim: {DIM})...")
    data = generate_clustered_data(N_VECTORS, DIM, n_clusters_real=500)
    queries = generate_clustered_data(N_QUERIES, DIM, n_clusters_real=10)

    # Note: We need a SimpleVectorDB just to hold the data pointers/calculate GT
    db_ref = myvector_db.SimpleVectorDB()
    print("Ingesting data into Reference DB (for Ground Truth)...")
    for i in range(N_VECTORS):
        db_ref.add_vector_numpy(data[i])

    # ---------------------------------------------------------
    # 2. ESTABLISH GROUND TRUTH (Brute Force)
    # ---------------------------------------------------------
    print(f"Calculating Ground Truth for {N_QUERIES} queries...")
    start_flat = time.time()
    ground_truth_results = []
    for i in range(N_QUERIES):
        res = db_ref.search(queries[i], K)
        ground_truth_results.append(res)

    time_flat = time.time() - start_flat
    avg_flat_ms = (time_flat / N_QUERIES) * 1000
    print(f"Avg Flat Search Latency: {avg_flat_ms:.4f} ms")

    # ---------------------------------------------------------
    # 3. ANNOY BENCHMARK LOOP
    # ---------------------------------------------------------
    print("\n--- ANNOY PERFORMANCE (Varying Tree Count) ---")
    header = f"{'TREES':<6} | {'BUILD TIME':<10} | {'AVG LATENCY (ms)':<18} | {'SPEEDUP':<10} | {'ACCURACY (%)':<12}"
    print("-" * len(header))
    print(header)
    print("-" * len(header))

    # To test different tree counts, we must rebuild the index each time.
    # (In production, you'd build 20 trees once and search subsets, but rebuilding is cleaner for benchmarks)
    for n_trees in TREE_COUNTS:

        # A. Build Index
        annoy = myvector_db.AnnoyIndex(DIM)

        # IMPORTANT: Annoy needs the raw data matrix.
        # We pass the list-of-lists (or numpy array if bound correctly).
        # Since our bindings take std::vector<std::vector<float>>,
        # we convert numpy -> list. (Optimizable later)
        t0 = time.time()
        annoy.build(data.tolist(), n_trees, K_LEAF)
        build_time = time.time() - t0

        # B. Run Search
        start_probe = time.time()
        annoy_results = []
        for i in range(N_QUERIES):
            # search_k is ignored in our greedy impl, pass dummy 0
            res = annoy.query(queries[i], K, 0)
            annoy_results.append(res)

        # C. Metrics
        total_time = time.time() - start_probe
        avg_time_ms = (total_time / N_QUERIES) * 1000
        speedup = avg_flat_ms / avg_time_ms
        accuracy = calculate_recall(ground_truth_results, annoy_results, K)

        print(
            f"{n_trees:<6} | {build_time:<10.2f} | {avg_time_ms:<18.4f} | {speedup:<10.1f} | {accuracy:<12.1f}"
        )

    print("-" * len(header))


if __name__ == "__main__":
    benchmark()
