"""Tests for FlatIndex (exact brute-force search)."""

import numpy as np
import pytest
from vegamdb import VegamDB


class TestFlatSearch:
    """Flat index should return exact nearest neighbors."""

    def test_exact_match(self, populated_db):
        """Querying with a stored vector should return itself as the closest."""
        db, data = populated_db
        results = db.search(data[0], k=1)
        assert results.ids[0] == 0
        assert results.distances[0] == pytest.approx(0.0, abs=1e-5)

    def test_k_results(self, populated_db):
        """Should return exactly k results."""
        db, data = populated_db
        results = db.search(data[0], k=5)
        assert len(results.ids) == 5
        assert len(results.distances) == 5

    def test_distances_sorted(self, populated_db):
        """Distances should be in ascending order."""
        db, data = populated_db
        results = db.search(data[0], k=10)
        for i in range(len(results.distances) - 1):
            assert results.distances[i] <= results.distances[i + 1]

    def test_unique_ids(self, populated_db):
        """All returned IDs should be unique."""
        db, data = populated_db
        results = db.search(data[0], k=10)
        assert len(set(results.ids)) == len(results.ids)

    def test_k_larger_than_db(self):
        """k > db.size() should return all available vectors."""
        db = VegamDB()
        data = np.random.RandomState(99).random((5, 16)).astype(np.float32)
        db.add_vector_numpy(data)
        results = db.search(data[0], k=100)
        assert len(results.ids) == 5

    def test_search_empty_db(self, db):
        """Searching an empty database should return empty results."""
        results = db.search([1.0, 2.0, 3.0], k=5)
        assert len(results.ids) == 0
        assert len(results.distances) == 0
