"""Tests for core VegamDB operations: add_vector, add_vector_numpy, size, dimension."""

import numpy as np
import pytest
from vegamdb import VegamDB


class TestAddVector:
    """Test adding vectors via Python list."""

    def test_add_single_vector(self, db):
        db.add_vector([1.0, 2.0, 3.0])
        assert db.size() == 1
        assert db.dimension() == 3

    def test_add_multiple_vectors(self, db):
        for i in range(100):
            db.add_vector([float(i)] * 10)
        assert db.size() == 100
        assert db.dimension() == 10


class TestAddVectorNumpy:
    """Test adding vectors via NumPy arrays."""

    def test_1d_array(self, db):
        vec = np.random.random(128).astype(np.float32)
        db.add_vector_numpy(vec)
        assert db.size() == 1
        assert db.dimension() == 128

    def test_2d_batch(self, db):
        data = np.random.random((500, 64)).astype(np.float32)
        db.add_vector_numpy(data)
        assert db.size() == 500
        assert db.dimension() == 64

    def test_2d_single_row(self, db):
        data = np.random.random((1, 32)).astype(np.float32)
        db.add_vector_numpy(data)
        assert db.size() == 1
        assert db.dimension() == 32

    def test_invalid_3d_raises(self, db):
        data = np.random.random((2, 3, 4)).astype(np.float32)
        with pytest.raises(RuntimeError):
            db.add_vector_numpy(data)

    def test_mixed_add(self, db):
        """Add via list, then via numpy, sizes accumulate."""
        db.add_vector([1.0, 2.0, 3.0])
        db.add_vector_numpy(np.array([4.0, 5.0, 6.0], dtype=np.float32))
        data = np.random.random((10, 3)).astype(np.float32)
        db.add_vector_numpy(data)
        assert db.size() == 12
        assert db.dimension() == 3


class TestEmptyDB:
    """Edge cases for empty databases."""

    def test_empty_size(self, db):
        assert db.size() == 0

    def test_empty_dimension(self, db):
        assert db.dimension() == 0
