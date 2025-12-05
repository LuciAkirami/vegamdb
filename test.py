import numpy as np
import myvector_db
import time

db = myvector_db.SimpleVectorDB()

# Increase size to make the difference obvious
num_vectors = 50000
dim = 128

print(f"Generating {num_vectors} vectors...")
data = np.random.random((num_vectors, dim)).astype(np.float32)

# --- Test 1: The Old Way (Slow) ---
print("Starting Standard List insertion...")
start = time.time()
for i in range(10000):  # Only do 10000 to show it works, otherwise it takes too long!
    db.add_vector(data[i])
print(f"Standard way (10000 items): {time.time() - start:.4f}s")

db = myvector_db.SimpleVectorDB()

# Increase size to make the difference obvious
num_vectors = 50000
dim = 128

# --- Test 2: The New Way (Fast) ---
print("Starting NumPy Direct insertion...")
start = time.time()
for i in range(num_vectors):
    # Pass the numpy row DIRECTLY. No .tolist()!
    db.add_vector_numpy(data[i])

print(f"NumPy way ({num_vectors} items): {time.time() - start:.4f}s")

print(f"Final DB Size: {db.get_size()}")

print(f"\n--- Testing Search ---")
# Create a dummy query (random vector)
query = np.random.random((dim)).astype(np.float32)

# Search for top 5 nearest neighbors
print("Searching for top 5 neighbors...")
start = time.time()
results = db.search(query, 5)  # <--- This will use stl.h implicit conversion
end = time.time()

print(f"Search time: {end - start:.4f}s")
print(f"Indices found: {results}")
