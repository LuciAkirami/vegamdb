import numpy as np
import myvector_db
import time
import os


def run_benchmark(n_vectors, dim=128):
    print(f"\n========================================")
    print(f"BENCHMARK: {n_vectors:,} Vectors (Dim: {dim})")
    print(f"========================================")

    # 1. Initialize
    db = myvector_db.VegamDB()

    # 2. Data Generation (Not measured in DB time)
    print("Generating data (NumPy)...")
    data = np.random.random((n_vectors, dim)).astype(np.float32)
    queries = np.random.random((100, dim)).astype(np.float32)  # 100 test queries

    # 3. Ingestion Test
    print("-> Testing Ingestion...")
    start = time.time()
    for i in range(n_vectors):
        db.add_vector_numpy(data[i])
    duration = time.time() - start
    print(f"   Ingestion Time: {duration:.4f}s")
    print(f"   Throughput:     {n_vectors / duration:,.0f} vectors/sec")

    # 4. Latency Test (Single Query)
    print("-> Testing Latency (1 Query)...")
    start = time.time()
    db.search(queries[0], 5)
    latency = time.time() - start
    print(f"   Latency:        {latency * 1000:.4f} ms")

    # 5. QPS Test (Throughput)
    # Run 100 queries back-to-back to see how well it handles load
    print(f"-> Testing QPS (100 Queries)...")
    start = time.time()
    for i in range(100):
        db.search(queries[i], 5)
    total_q_time = time.time() - start
    avg_q_time = total_q_time / 100
    qps = 1 / avg_q_time
    print(f"   Total Time:     {total_q_time:.4f}s")
    print(f"   Avg Latency:    {avg_q_time * 1000:.4f} ms")
    print(f"   QPS:            {qps:,.2f} queries/sec")

    # 6. Disk I/O Test
    filename = f"bench_temp_{n_vectors}.bin"
    print("-> Testing Save/Load...")

    start = time.time()
    db.save(filename)
    save_time = time.time() - start
    print(f"   Save Time:      {save_time:.4f}s")

    # Cleanup
    if os.path.exists(filename):
        os.remove(filename)

    return latency  # Return latency to track scaling


if __name__ == "__main__":
    # We define levels of difficulty
    # Level 1: 10,000 (Toy)
    # Level 2: 100,000 (Small App)
    # Level 3: 1,000,000 (Production Scale - 128MB RAM approx)

    levels = [10_000, 100_000, 500_000]

    # NOTE: 1 Million might take ~1 second per search.
    # Uncomment next line if you feel brave!
    levels.append(1_000_000)

    for n in levels:
        run_benchmark(n)
