"""Tests for Annoy Index (approximate search with random projection trees)."""

import numpy as np
import pytest
from vegamdb import VegamDB, AnnoyIndexParams


@pytest.fixture
def annoy_db():
    """VegamDB with Annoy index built on 1000 vectors."""
    db = VegamDB()
    data = np.random.RandomState(42).random((1000, 64)).astype(np.float32)
    db.add_vector_numpy(data)
    db.use_annoy_index(num_trees=10, k_leaf=50)
    db.build_index()
    return db, data


class TestAnnoyIndex:
    """Annoy index search tests."""

    def test_search_returns_results(self, annoy_db):
        db, data = annoy_db
        results = db.search(data[0], k=5)
        assert len(results.ids) == 5
        assert len(results.distances) == 5

    def test_distances_sorted(self, annoy_db):
        db, data = annoy_db
        results = db.search(data[0], k=10)
        for i in range(len(results.distances) - 1):
            assert results.distances[i] <= results.distances[i + 1]

    def test_unique_ids(self, annoy_db):
        db, data = annoy_db
        results = db.search(data[0], k=10)
        assert len(set(results.ids)) == len(results.ids)

    def test_search_k_override(self, annoy_db):
        """Higher search_k should give same or better results."""
        db, data = annoy_db
        query = data[0]

        params_low = AnnoyIndexParams()
        params_low.search_k = 50
        params_low.use_priority_queue = True
        results_low = db.search(query, k=5, params=params_low)

        params_high = AnnoyIndexParams()
        params_high.search_k = 500
        params_high.use_priority_queue = True
        results_high = db.search(query, k=5, params=params_high)

        assert len(results_low.ids) == 5
        assert len(results_high.ids) == 5
        assert results_high.distances[0] <= results_low.distances[0]


class TestAnnoyStrategy:
    """Test greedy vs priority queue strategy toggle."""

    def test_greedy_via_constructor(self):
        """use_priority_queue=False at index creation uses greedy search."""
        db = VegamDB()
        data = np.random.RandomState(42).random((1000, 64)).astype(np.float32)
        db.add_vector_numpy(data)
        db.use_annoy_index(num_trees=10, k_leaf=50, use_priority_queue=False)
        db.build_index()
        results = db.search(data[0], k=5)
        assert len(results.ids) == 5

    def test_greedy_via_params(self, annoy_db):
        """use_priority_queue=False can be set per-query via params."""
        db, data = annoy_db
        params = AnnoyIndexParams()
        params.search_k = 500
        params.use_priority_queue = False
        results = db.search(data[0], k=5, params=params)
        assert len(results.ids) == 5

    def test_both_strategies_find_exact_match(self, annoy_db):
        """Both strategies should find the query vector itself."""
        db, data = annoy_db
        query = data[0]

        # Priority queue
        params_pq = AnnoyIndexParams()
        params_pq.search_k = 500
        params_pq.use_priority_queue = True
        results_pq = db.search(query, k=1, params=params_pq)

        # Greedy
        params_greedy = AnnoyIndexParams()
        params_greedy.search_k = 500
        params_greedy.use_priority_queue = False
        results_greedy = db.search(query, k=1, params=params_greedy)

        assert results_pq.ids[0] == 0
        assert results_greedy.ids[0] == 0
