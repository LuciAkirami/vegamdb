import numpy as np
import myvector_db
import time
import os


def test_persistence():
    filename = "test_ivf_index.bin"
    N = 10000
    DIM = 128
    CLUSTERS = 50
    QUERY_K = 10
    NPROBE = 5

    print("--- 1. SETUP & TRAINING ---")
    # Generate random data
    print(f"Generating {N} vectors...")
    data = np.random.random((N, DIM)).astype(np.float32)
    query = np.random.random((DIM)).astype(np.float32)

    # Init DB 1
    db_orig = myvector_db.SimpleVectorDB()
    for i in range(N):
        db_orig.add_vector_numpy(data[i])

    # Build Index
    print("Building Index...")
    db_orig.build_index(CLUSTERS, 10)

    # Run a Search BEFORE saving to establish a baseline
    print("Running reference search on original DB...")
    results_orig = db_orig.search_ivf(query, QUERY_K, NPROBE)
    print(f"Original Results (First 5): {results_orig[:5]}")

    # Save
    print(f"Saving to {filename}...")
    db_orig.save(filename)

    # Delete DB 1 object to ensure we aren't accidentally reading from memory
    del db_orig

    print("\n--- 2. LOADING & VERIFICATION ---")
    db_loaded = myvector_db.SimpleVectorDB()
    db_loaded.load(filename)

    # TEST 1: Size Verification
    size = db_loaded.get_size()
    print(f"Loaded DB Size: {size}")
    if size != N:
        print("FAILURE: Size mismatch.")
        return
    else:
        print("SUCCESS: Size matches.")

    # TEST 2: Data Integrity (Spot Check)
    # We grab a specific vector from the source data (e.g., index 5000)
    # and verify it exists at index 5000 in the loaded DB.
    # We use Brute Force search with K=1 to find the exact match (distance should be 0)
    check_idx = 5000
    check_vec = data[check_idx]
    print(f"Verifying data integrity for vector index {check_idx}...")

    # Search for the vector itself
    search_res = db_loaded.search(check_vec, 1)
    found_idx = search_res[0]

    if found_idx == check_idx:
        print("SUCCESS: Retrieved exact vector from loaded data (Data Integrity OK).")
    else:
        print(f"FAILURE: Expected index {check_idx}, got {found_idx}")

    # TEST 3: IVF Search Consistency
    # Does the index structure work exactly the same?
    print("Running search on loaded DB with same query...")
    results_loaded = db_loaded.search_ivf(query, QUERY_K, NPROBE)
    print(f"Loaded Results (First 5):   {results_loaded[:5]}")

    if results_orig == results_loaded:
        print("SUCCESS: Search results are identical before and after load.")
    else:
        print("FAILURE: Search results differ.")
        print(f"Orig:   {results_orig}")
        print(f"Loaded: {results_loaded}")

    # Cleanup
    if os.path.exists(filename):
        os.remove(filename)


if __name__ == "__main__":
    test_persistence()
