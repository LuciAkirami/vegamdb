import numpy as np
import myvector_db
import time

# 1. Initialize the Database
db = myvector_db.SimpleVectorDB()
print("Database initialized successfully!")

# 2. Generate some dummy data using NumPy
# We create 1000 vectors, each with dimension 128 (typical for small models)
num_vectors = 1000
dim = 128

print(f"Generating {num_vectors} vectors of dimension {dim}...")
# Create random float data
data = np.random.random((num_vectors, dim)).astype(np.float32)

# 3. Add vectors to the C++ DB
start_time = time.time()
for i in range(num_vectors):
    # We must convert the numpy row to a standard list for now
    # (We will optimize this later!)
    db.add_vector(data[i].tolist())

end_time = time.time()

# 4. Verify
print(f"Successfully stored {db.get_size()} vectors.")
print(f"Time taken: {end_time - start_time:.4f} seconds")
