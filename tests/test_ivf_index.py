"""Tests for IVF Index (approximate search with clustering)."""

import numpy as np
import pytest
from vegamdb import VegamDB, IVFSearchParams


@pytest.fixture
def ivf_db():
    """VegamDB with IVF index built on 1000 vectors."""
    db = VegamDB()
    data = np.random.RandomState(42).random((1000, 64)).astype(np.float32)
    db.add_vector_numpy(data)
    db.use_ivf_index(n_clusters=10, max_iters=50, n_probe=1)
    db.build_index()
    return db, data


class TestIVFIndex:
    """IVF index search tests."""

    def test_search_returns_results(self, ivf_db):
        db, data = ivf_db
        results = db.search(data[0], k=5)
        assert len(results.ids) == 5
        assert len(results.distances) == 5

    def test_distances_sorted(self, ivf_db):
        db, data = ivf_db
        results = db.search(data[0], k=10)
        for i in range(len(results.distances) - 1):
            assert results.distances[i] <= results.distances[i + 1]

    def test_unique_ids(self, ivf_db):
        db, data = ivf_db
        results = db.search(data[0], k=10)
        assert len(set(results.ids)) == len(results.ids)

    def test_nprobe_override(self, ivf_db):
        """Higher n_probe should return same or better quality results."""
        db, data = ivf_db
        query = data[0]

        params_low = IVFSearchParams()
        params_low.n_probe = 1
        results_low = db.search(query, k=5, params=params_low)

        params_high = IVFSearchParams()
        params_high.n_probe = 10
        results_high = db.search(query, k=5, params=params_high)

        # Both should return valid results
        assert len(results_low.ids) == 5
        assert len(results_high.ids) == 5
        # Higher n_probe should find at least as good a nearest neighbor
        assert results_high.distances[0] <= results_low.distances[0]
