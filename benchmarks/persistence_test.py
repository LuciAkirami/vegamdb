import numpy as np
import vegamdb
import os
import time


def test_flat_persistence():
    """Test save/load with FlatIndex (default, no explicit index set)."""
    filename = "test_flat.bin"
    N = 1000
    DIM = 64
    K = 5

    print("=== FLAT INDEX PERSISTENCE ===")

    data = np.random.random((N, DIM)).astype(np.float32)
    query = data[42]  # Use a known vector as query

    # --- Save ---
    db = vegamdb.VegamDB()
    for i in range(N):
        db.add_vector_numpy(data[i])

    results_before = db.search(query, K)
    print(f"  Results before save: {results_before.ids[:5]}")

    t0 = time.time()
    db.save(filename)
    print(f"  Save time: {time.time() - t0:.4f}s")
    del db

    # --- Load ---
    db2 = vegamdb.VegamDB()
    t0 = time.time()
    db2.load(filename)
    print(f"  Load time: {time.time() - t0:.4f}s")

    assert db2.size() == N, f"FAIL: Size mismatch {db2.size()} != {N}"
    print(f"  Size after load: {db2.size()}")

    results_after = db2.search(query, K)
    print(f"  Results after load:  {results_after.ids[:5]}")

    assert results_before.ids == results_after.ids, "FAIL: Results differ!"
    print("  Results match")

    os.remove(filename)
    print("  PASSED\n")


def test_ivf_persistence():
    """Test save/load with IVFIndex."""
    filename = "test_ivf.bin"
    N = 10000
    DIM = 128
    K = 10
    NPROBE = 5

    print("=== IVF INDEX PERSISTENCE ===")

    data = np.random.random((N, DIM)).astype(np.float32)
    query = np.random.random(DIM).astype(np.float32)

    # --- Build & Save ---
    db = vegamdb.VegamDB()
    for i in range(N):
        db.add_vector_numpy(data[i])

    db.use_ivf_index(n_clusters=50, max_iters=10)
    db.build_index()

    params = vegamdb.IVFSearchParams()
    params.n_probe = NPROBE
    results_before = db.search(query, K, params)
    print(f"  Results before save: {results_before.ids[:5]}")

    t0 = time.time()
    db.save(filename)
    save_time = time.time() - t0
    print(f"  Save time: {save_time:.4f}s")
    del db

    # --- Load ---
    db2 = vegamdb.VegamDB()
    t0 = time.time()
    db2.load(filename)
    print(f"  Load time: {time.time() - t0:.4f}s")

    assert db2.size() == N, f"FAIL: Size mismatch {db2.size()} != {N}"
    print(f"  Size after load: {db2.size()}")

    params2 = vegamdb.IVFSearchParams()
    params2.n_probe = NPROBE
    results_after = db2.search(query, K, params2)
    print(f"  Results after load:  {results_after.ids[:5]}")

    assert results_before.ids == results_after.ids, "FAIL: Results differ!"
    print("  Results match")

    # Verify data integrity: search for a known vector
    check_vec = data[5000]
    spot_check = db2.search(check_vec, 1)
    assert spot_check.ids[0] == 5000, f"FAIL: Expected 5000, got {spot_check.ids[0]}"
    print("  Data integrity")

    os.remove(filename)
    print("  PASSED\n")


def test_annoy_persistence():
    """Test save/load with AnnoyIndex."""
    filename = "test_annoy.bin"
    N = 10000
    DIM = 64
    K = 10

    print("=== ANNOY INDEX PERSISTENCE ===")

    data = np.random.random((N, DIM)).astype(np.float32)
    query = np.random.random(DIM).astype(np.float32)

    # --- Build & Save ---
    db = vegamdb.VegamDB()
    for i in range(N):
        db.add_vector_numpy(data[i])

    db.use_annoy_index(num_trees=10, k_leaf=50)
    db.build_index()

    results_before = db.search(query, K)
    print(f"  Results before save: {results_before.ids[:5]}")

    t0 = time.time()
    db.save(filename)
    print(f"  Save time: {time.time() - t0:.4f}s")
    del db

    # --- Load ---
    db2 = vegamdb.VegamDB()
    t0 = time.time()
    db2.load(filename)
    print(f"  Load time: {time.time() - t0:.4f}s")

    assert db2.size() == N, f"FAIL: Size mismatch {db2.size()} != {N}"
    print(f"  Size after load: {db2.size()}")

    results_after = db2.search(query, K)
    print(f"  Results after load:  {results_after.ids[:5]}")

    assert results_before.ids == results_after.ids, "FAIL: Results differ!"
    print("  Results match")

    os.remove(filename)
    print("  PASSED\n")


if __name__ == "__main__":
    test_flat_persistence()
    test_ivf_persistence()
    test_annoy_persistence()
    print("ALL PERSISTENCE TESTS PASSED!")
