import numpy as np
import vegamdb
import time

db = vegamdb.VegamDB()

# Increase size to make the difference obvious
num_vectors = 50000
dim = 128

print(f"Generating {num_vectors} vectors...")
data = np.random.random((num_vectors, dim)).astype(np.float32)

# --- Test 1: The Old Way (Slow) ---
print("Starting Standard List insertion...")
start = time.time()
for i in range(10000):  # Only do 10000 to show it works, otherwise it takes too long!
    db.add_vector(data[i].tolist())
print(f"Standard way (10000 items): {time.time() - start:.4f}s")

db = vegamdb.VegamDB()

# --- Test 2: The New Way (Fast) ---
print("Starting NumPy Direct insertion...")
start = time.time()
for i in range(num_vectors):
    # Pass the numpy row DIRECTLY. No .tolist()!
    db.add_vector_numpy(data[i])

print(f"NumPy way ({num_vectors} items): {time.time() - start:.4f}s")

print(f"Final DB Size: {db.size()}")

print(f"\n--- Testing Search ---")
# Create a dummy query (random vector)
query = np.random.random((dim)).astype(np.float32)

# Search for top 5 nearest neighbors
print("Searching for top 5 neighbors...")
start = time.time()
results = db.search(query, 5)
end = time.time()

print(f"Search time: {end - start:.4f}s")
print(f"Indices found: {results.ids}")

print(f"\n--- Testing Search (Verification with Numpy) ---")
# Create a random query
query = np.random.random((dim)).astype(np.float32)

# 1. Run C++ Search
print("Searching with C++...")
start = time.time()
results = db.search(query, 5)
end = time.time()
print(f"C++ Time: {end - start:.4f}s")
print(f"C++ Indices: {results.ids}")

# 2. Verify with NumPy (Ground Truth)
# We manually calculate distance for comparison
print("\nVerifying with NumPy...")
# Calculate Euclidean distance for ALL vectors in 'data' against 'query'
# axis=1 means sum across the columns (dimensions)
start = time.time()
diff = data - query
dists = np.linalg.norm(diff, axis=1)

# Get the indices of the smallest 5 distances
# argsort returns the indices that would sort the array
numpy_indices = np.argsort(dists)[:5]
end = time.time()

print(f"Numpy Time: {end - start:.4f}s")
print(f"NumPy Indices: {numpy_indices.tolist()}")

# 3. Compare
# We use set() because the order of TIES might differ, but the set should be identical.
if set(results.ids) == set(numpy_indices):
    print("\nSUCCESS: C++ results match NumPy Ground Truth!")
else:
    print("\nERROR: Results do not match!")

print(f"\n--- Testing Persistence (Save/Load) ---")
filename = "my_index.bin"

# 1. Save
print(f"Saving {db.size()} vectors to {filename}...")
start = time.time()
db.save(filename)
print(f"Save time: {time.time() - start:.4f}s")

# 2. Kill the DB (Simulate restarting the app)
print("Deleting Database object...")
del db
db = vegamdb.VegamDB()
print(f"New DB Size: {db.size()} (Should be 0)\n\n")

# 3. Load
print(f"Loading from {filename}...")
start = time.time()
db.load(filename)
print(f"Load time: {time.time() - start:.4f}s")

# 4. Verify
print(f"Restored DB Size: {db.size()}")
if db.size() == 50000:
    print("SUCCESS: Persistence working!")
else:
    print("ERROR: Data lost!")
