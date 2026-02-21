"""Tests for save/load persistence across all index types."""

import os
import numpy as np
import pytest
from vegamdb import VegamDB


@pytest.fixture
def tmp_path_db(tmp_path):
    """Returns a temp file path for save/load tests."""
    return str(tmp_path / "test_db.vegam")


class TestFlatPersistence:
    """Save/load with flat index."""

    def test_round_trip(self, tmp_path_db):
        db = VegamDB()
        data = np.random.RandomState(42).random((100, 32)).astype(np.float32)
        db.add_vector_numpy(data)

        query = data[0]
        results_before = db.search(query, k=5)

        db.save(tmp_path_db)
        assert os.path.exists(tmp_path_db)

        db2 = VegamDB()
        db2.load(tmp_path_db)

        assert db2.size() == 100
        assert db2.dimension() == 32

        results_after = db2.search(query, k=5)
        assert results_before.ids == results_after.ids
        assert results_before.distances == pytest.approx(
            results_after.distances, abs=1e-5
        )


class TestIVFPersistence:
    """Save/load with IVF index."""

    def test_round_trip(self, tmp_path_db):
        db = VegamDB()
        data = np.random.RandomState(42).random((500, 32)).astype(np.float32)
        db.add_vector_numpy(data)
        db.use_ivf_index(n_clusters=5, max_iters=50, n_probe=3)
        db.build_index()

        query = data[0]
        results_before = db.search(query, k=5)

        db.save(tmp_path_db)

        db2 = VegamDB()
        db2.load(tmp_path_db)

        assert db2.size() == 500

        results_after = db2.search(query, k=5)
        assert results_before.ids == results_after.ids
        assert results_before.distances == pytest.approx(
            results_after.distances, abs=1e-5
        )


class TestAnnoyPersistence:
    """Save/load with Annoy index."""

    def test_round_trip(self, tmp_path_db):
        db = VegamDB()
        data = np.random.RandomState(42).random((500, 32)).astype(np.float32)
        db.add_vector_numpy(data)
        db.use_annoy_index(num_trees=5, k_leaf=50)
        db.build_index()

        query = data[0]
        results_before = db.search(query, k=5)

        db.save(tmp_path_db)

        db2 = VegamDB()
        db2.load(tmp_path_db)

        assert db2.size() == 500

        results_after = db2.search(query, k=5)
        assert results_before.ids == results_after.ids
        assert results_before.distances == pytest.approx(
            results_after.distances, abs=1e-5
        )
