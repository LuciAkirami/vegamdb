import numpy as np
from vegamdb import VegamDB

db = VegamDB()

# Test 2D batch
data = np.random.random((1000, 128)).astype(np.float32)
db.add_vector_numpy(data)
print(f"After 2D batch: {db.size()} vectors")  # Should be 1000
print("Dimension", db.dimension())

# Test 1D single
single = np.random.random(128).astype(np.float32)
db.add_vector_numpy(single)
print(f"After 1D single: {db.size()} vectors")  # Should be 1001

# Test search still works
results = db.search(single, k=5)
print(f"Search results: {results.ids}")
