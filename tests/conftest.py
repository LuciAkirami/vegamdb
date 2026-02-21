"""Shared test fixtures for VegamDB test suite."""

import numpy as np
import pytest
from vegamdb import VegamDB


@pytest.fixture
def db():
    """Empty VegamDB instance."""
    return VegamDB()


@pytest.fixture
def populated_db():
    """VegamDB with 1000 vectors of dim 64."""
    db = VegamDB()
    data = np.random.RandomState(42).random((1000, 64)).astype(np.float32)
    db.add_vector_numpy(data)
    return db, data
