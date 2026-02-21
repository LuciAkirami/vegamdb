# vegamdb/_vegamdb.pyi
#
# Type stubs for the compiled C++ extension module.
# Provides IDE autocomplete and type checking support.

from typing import List, Optional

import numpy


class SearchResults:
    """Container returned by VegamDB.search().

    Attributes:
        ids: Indices of the nearest neighbors in insertion order.
        distances: Corresponding distances to the query vector.
    """

    ids: List[int]
    """List of integer indices of the nearest neighbors."""
    distances: List[float]
    """List of distances corresponding to each neighbor."""


class SearchParams:
    """Base class for index-specific search parameters."""

    ...


class IVFSearchParams(SearchParams):
    """Search parameters for IVF index.

    Attributes:
        n_probe: Number of clusters to probe during search.
            Higher values improve recall at the cost of speed. Default: 1.

    Example::

        params = IVFSearchParams()
        params.n_probe = 10
        results = db.search(query, k=5, params=params)
    """

    n_probe: int
    """Number of clusters to probe during search (default: 1)."""
    def __init__(self) -> None: ...


# TODO: Uncomment once search_k_nodes runtime override is implemented
# class AnnoyIndexParams(SearchParams):
#     """Search parameters for Annoy index.
#
#     Attributes:
#         search_k_nodes: Number of leaf nodes to inspect during search.
#             Higher values improve recall at the cost of speed.
#
#     Example::
#
#         params = AnnoyIndexParams()
#         params.search_k_nodes = 50
#         results = db.search(query, k=5, params=params)
#     """
#
#     search_k_nodes: int
#     """Number of leaf nodes to inspect during search."""
#     def __init__(self) -> None: ...


class IndexBase:
    """Abstract base class for all index types."""

    ...


class FlatIndex(IndexBase):
    """Brute-force flat index for exact nearest neighbor search."""

    def __init__(self) -> None: ...


class IVFIndex(IndexBase):
    """Inverted File Index using K-Means clustering for approximate search."""

    def __init__(
        self,
        n_clusters: int,
        dimension: int,
        max_iters: int = 50,
        n_probe: int = 1,
    ) -> None: ...


class AnnoyIndex(IndexBase):
    """Approximate Nearest Neighbors using random projection trees."""

    def __init__(
        self,
        dimension: int,
        num_trees: int,
        k_leaf: int,
        search_k_nodes: int = 1,
    ) -> None: ...


class VegamDB:
    """A high-performance vector database with pluggable index types."""

    def __init__(self) -> None:
        """Create a new empty VegamDB instance."""
        ...

    def dimension(self) -> int:
        """Return the dimensionality of stored vectors (0 if empty)."""
        ...

    def add_vector(self, vec: List[float]) -> None:
        """Add a single vector as a Python list of floats."""
        ...

    def add_vector_numpy(self, input_array: numpy.ndarray) -> None:
        """Add a single vector from a 1D NumPy float32 array (zero-copy)."""
        ...

    def size(self) -> int:
        """Return the number of vectors stored in the database."""
        ...

    def use_flat_index(self) -> None:
        """Set the index to brute-force flat search (exact, no training needed)."""
        ...

    def use_ivf_index(
        self, n_clusters: int, max_iters: int = 50, n_probe: int = 1
    ) -> None:
        """Set the index to IVF (Inverted File Index) for approximate search.

        Args:
            n_clusters: Number of Voronoi cells (clusters) for partitioning.
            max_iters: Maximum K-Means iterations for training (default: 50).
            n_probe: Number of clusters to search at query time (default: 1).
        """
        ...

    def use_annoy_index(
        self, num_trees: int, k_leaf: int, search_k_nodes: int = 1
    ) -> None:
        """Set the index to Annoy (Approximate Nearest Neighbors Oh Yeah).

        Args:
            num_trees: Number of random projection trees to build.
            k_leaf: Maximum number of points in each leaf node.
            search_k_nodes: Number of leaf nodes to search at query time (default: 1).
        """
        ...

    def build_index(self) -> None:
        """Explicitly build/train the current index on stored vectors."""
        ...

    def search(
        self,
        query: List[float],
        k: int,
        params: Optional[SearchParams] = None,
    ) -> SearchResults:
        """Search for the k nearest neighbors of a query vector.

        Args:
            query: 1D list of floats representing the query vector.
            k: Number of nearest neighbors to return.
            params: Optional IVFSearchParams or AnnoyIndexParams.

        Returns:
            SearchResults with .ids (list[int]) and .distances (list[float]).
        """
        ...

    def save(self, filename: str) -> None:
        """Save the database (vectors + index) to a binary file."""
        ...

    def load(self, filename: str) -> None:
        """Load a database (vectors + index) from a binary file."""
        ...


class KMeansIndex:
    """Result container for K-Means training."""

    centroids: List[List[float]]
    """List of cluster centroid vectors."""
    buckets: List[List[int]]
    """List of clusters, each containing vector indices."""


class KMeans:
    """Standalone K-Means clustering utility."""

    def __init__(self, n_clusters: int, dimension: int, max_iters: int) -> None:
        """Create a KMeans instance with given parameters."""
        ...

    def train(self, data: List[List[float]]) -> KMeansIndex:
        """Train K-Means on the provided data and return a KMeansIndex."""
        ...
