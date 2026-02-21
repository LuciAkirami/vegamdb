# VegamDB

A high-performance vector database written in C++ with Python bindings. VegamDB provides fast nearest neighbor search with pluggable index types, zero-copy NumPy integration, and built-in persistence.

## Features

- **Multiple Index Types** -- Flat (exact brute-force), IVF (inverted file with K-Means), and Annoy (random projection trees)
- **C++ Core** -- All indexing and search logic runs in optimized C++17 with `-O3` and `-march=native`
- **Zero-Copy NumPy** -- Vectors pass directly from NumPy arrays to C++ via pointer, with no intermediate copies
- **Persistence** -- Save and load the entire database (vectors + index) to a single binary file
- **Pluggable Architecture** -- Switch index types at runtime without changing application code
- **Type-Safe Python API** -- Full type stubs (`.pyi`) for IDE autocomplete and static analysis

## Installation

### From Source

```bash
git clone https://github.com/LuciAkirami/vegamdb.git
cd vegamdb
pip install .
```

### Requirements

- Python >= 3.8
- CMake >= 3.15
- A C++17 compatible compiler (GCC 7+, Clang 5+, MSVC 2017+)
- NumPy

### Development Install

For development, use an editable install so changes to Python files take effect immediately:

```bash
pip install scikit-build-core pybind11 numpy
pip install -e . --no-build-isolation
```

## Quick Start

```python
import numpy as np
from vegamdb import VegamDB

# Create a database
db = VegamDB()

# Add vectors (batch — pass a 2D NumPy array)
data = np.random.random((10000, 128)).astype(np.float32)
db.add_vector_numpy(data)

# Search (defaults to exact flat search)
query = np.random.random(128).astype(np.float32)
results = db.search(query, k=5)

print(results.ids)        # [4823, 1092, 7744, 331, 5619]
print(results.distances)  # [4.12, 4.15, 4.18, 4.21, 4.23]
```

## Index Types

VegamDB supports three index types, each offering a different trade-off between speed and accuracy.

### Flat Index (Default)

Exact brute-force search. Computes the Euclidean distance between the query and every stored vector. Always returns the true nearest neighbors.

```python
db.use_flat_index()
results = db.search(query, k=10)
```

| Metric       | Value        |
| ------------ | ------------ |
| Accuracy     | 100%         |
| Build Time   | None         |
| Best For     | Small datasets (< 50K vectors), ground truth validation |

### IVF Index (Inverted File)

Partitions vectors into clusters using K-Means. At query time, only the closest clusters are searched, trading some accuracy for a large speedup.

```python
db.use_ivf_index(n_clusters=100, max_iters=20, n_probe=1)
db.build_index()

# Search with custom probe count
from vegamdb import IVFSearchParams
params = IVFSearchParams()
params.n_probe = 10  # Search 10 of 100 clusters
results = db.search(query, k=10, params=params)
```

| Parameter     | Description                                      | Default |
| ------------- | ------------------------------------------------ | ------- |
| `n_clusters`  | Number of Voronoi cells (partitions)             | --      |
| `max_iters`   | Maximum K-Means training iterations              | 50      |
| `n_probe`     | Clusters to search at query time                 | 1       |

### Annoy Index (Approximate Nearest Neighbors)

Builds a forest of random projection trees. Each tree recursively splits the vector space with random hyperplanes. At query time, multiple trees are traversed to collect candidate neighbors.

```python
db.use_annoy_index(num_trees=10, k_leaf=50)
db.build_index()

results = db.search(query, k=10)
```

| Parameter        | Description                                    | Default |
| ---------------- | ---------------------------------------------- | ------- |
| `num_trees`      | Number of random projection trees              | --      |
| `k_leaf`         | Maximum points per leaf node                   | --      |

### Choosing an Index

| Use Case                     | Recommended Index | Why                                   |
| ---------------------------- | ----------------- | ------------------------------------- |
| Small dataset (< 50K)        | Flat              | Exact results, no training overhead   |
| Medium dataset (50K - 1M)    | IVF               | Good speed/accuracy with tunable probe|
| Large dataset (1M+)          | Annoy             | Fast tree traversal, low memory       |
| Ground truth / benchmarking  | Flat              | Guaranteed correct results            |

## Persistence

Save and load the entire database state, including vectors and the trained index:

```python
# Save
db.save("my_database.bin")

# Load into a fresh instance
db2 = VegamDB()
db2.load("my_database.bin")

assert db2.size() == db.size()
```

The index type and its trained state are serialized automatically. After loading, the index is ready to search without rebuilding.

## API Reference

### VegamDB

| Method                 | Description                                                       |
| ---------------------- | ----------------------------------------------------------------- |
| `VegamDB()`            | Create a new empty database instance                              |
| `add_vector(vec)`      | Add a vector from a Python list of floats                         |
| `add_vector_numpy(arr)`| Add vectors from a 1D `(dim,)` or 2D `(n, dim)` NumPy array      |
| `size()`               | Return the number of stored vectors                               |
| `dimension()`          | Return the dimensionality of stored vectors (0 if empty)          |
| `use_flat_index()`     | Set index to brute-force flat search                              |
| `use_ivf_index(...)`   | Set index to IVF with specified cluster configuration             |
| `use_annoy_index(...)` | Set index to Annoy with specified tree configuration              |
| `build_index()`        | Explicitly build/train the current index                          |
| `search(query, k, params=None)` | Search for k nearest neighbors, returns `SearchResults` |
| `save(filename)`       | Save database and index to a binary file                          |
| `load(filename)`       | Load database and index from a binary file                        |

### SearchResults

| Attribute    | Type          | Description                                      |
| ------------ | ------------- | ------------------------------------------------ |
| `ids`        | `list[int]`   | Indices of nearest neighbors (insertion order)   |
| `distances`  | `list[float]` | Euclidean distances to the query vector           |

### Search Parameters

**IVFSearchParams** -- Override the default probe count for IVF search:
- `n_probe` (int): Number of clusters to search. Higher values improve recall at the cost of latency.

## Architecture

```
                        VegamDB (Orchestrator)
                       /                      \
               VectorStore                  IndexBase
            (raw float vectors)           (search strategy)
                                         /      |       \
                                     Flat      IVF     Annoy
                                   (exact)  (K-Means)  (trees)
```

- **VegamDB** -- Main entry point. Manages the vector store and delegates search to the active index.
- **VectorStore** -- Stores raw vectors in a `vector<vector<float>>`. Handles serialization.
- **IndexBase** -- Abstract interface that all index types implement (`build`, `search`, `save`, `load`).
- **FlatIndex** -- Iterates over all vectors, computing Euclidean distance. O(n) per query.
- **IVFIndex** -- Trains K-Means centroids, assigns vectors to clusters, searches only nearby clusters.
- **AnnoyIndex** -- Builds a forest of binary trees using random hyperplane splits for fast traversal.

## Project Structure

```
vegamdb/
├── include/                  # C++ headers
│   ├── VegamDB.hpp
│   ├── indexes/              # IndexBase, FlatIndex, IVFIndex, AnnoyIndex, KMeans
│   ├── storage/              # VectorStore
│   └── utils/                # Math utilities (Euclidean distance, dot product)
├── src/                      # C++ implementation
│   ├── VegamDB.cpp
│   ├── bindings.cpp          # pybind11 Python bindings
│   ├── indexes/
│   ├── storage/
│   └── utils/
├── vegamdb/                  # Python package
│   ├── __init__.py           # Public API re-exports
│   └── _vegamdb.pyi          # Type stubs for IDE support
├── benchmarks/               # Performance benchmarks
├── CMakeLists.txt            # C++ build configuration
└── pyproject.toml            # Python packaging (scikit-build-core)
```

## Benchmarks

Run the included benchmarks to evaluate performance on your hardware:

```bash
# Stress test (Flat index, varying dataset sizes)
python benchmarks/stress_test.py

# IVF benchmark (accuracy vs speed trade-off across probe counts)
python benchmarks/ivf_benchmarks.py

# Annoy benchmark (accuracy vs speed trade-off across tree counts)
python benchmarks/annoy_benchmark.py
```

## License

MIT
