import contextlib
import os
import sys
from typing import List, Optional
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import numpy as np

# =========================================================
# 1. SETUP PATHS & IMPORT C++ LIBRARY
# =========================================================
# We need to look one folder up ("../") to find myvector_db.so
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

import myvector_db

# =========================================================
# 2. GLOBAL STATE & CONFIG
# =========================================================
# The database instance will live here.
# DB: Optional[myvector_db.SimpleVectorDB] = None
INDEX_FILE = os.path.join(parent_dir, "saved_index.bin")

# =========================================================
# 3. PYDANTIC MODELS (The API Contract)
# =========================================================


# Input for adding a vector
class VectorInput(BaseModel):
    vector: List[float]  # User sends [0.1, 0.2, ...]


# Input for searching
class SearchInput(BaseModel):
    query: List[float]
    k: int = 5
    nprobe: int = 10  # Default to balanced accuracy


# Input for training
class TrainInput(BaseModel):
    num_clusters: int = 100
    max_iters: int = 10


# =========================================================
# 4. LIFESPAN (Startup & Shutdown)
# =========================================================
@contextlib.asynccontextmanager
async def lifespan(app: FastAPI):
    global DB
    print("--- SERVER STARTUP ---")

    # 1. Initialize C++ Object
    DB = myvector_db.SimpleVectorDB()

    # 2. Try to load existing data
    if os.path.exists(INDEX_FILE):
        print(f"Loading data from {INDEX_FILE}...")
        try:
            DB.load(INDEX_FILE)
            print(f"Loaded {DB.get_size()} vectors.")
        except Exception as e:
            print(f"Error loading index: {e}")
    else:
        print("No existing index found. Starting fresh.")

    yield  # Server runs here...

    print("--- SERVER SHUTDOWN ---")
    # Optional: Auto-save on shutdown?
    # DB.save(INDEX_FILE)


# Initialize App
app = FastAPI(lifespan=lifespan)

# =========================================================
# 5. API ENDPOINTS
# =========================================================


@app.get("/")
def health_check():
    """Returns the status and size of the DB."""
    if not DB:
        return {"status": "error", "message": "DB not initialized"}
    return {"status": "online", "vectors_stored": DB.get_size()}


@app.post("/vectors")
def add_vector(item: VectorInput):
    """
    Adds a single vector to the database.
    Converts Python List -> NumPy Array -> C++ Pointer.
    """
    # 1. Convert List to NumPy (Float32 is crucial!)
    vec_np = np.array(item.vector, dtype=np.float32)

    # 2. Pass to C++ (Zero-copy logic)
    DB.add_vector_numpy(vec_np)

    return {"message": "Vector added", "total": DB.get_size()}


@app.post("/index")
def train_index(config: TrainInput):
    """
    Triggers K-Means training (IVF Index).
    WARNING: This blocks the thread while training!
    """
    if DB.get_size() < config.num_clusters:
        raise HTTPException(status_code=400, detail="Not enough data to train clusters")

    print(f"Starting Training: {config.num_clusters} clusters...")

    # Call C++ build_index
    DB.build_index(config.num_clusters, config.max_iters)

    return {"message": "Training Complete", "clusters": config.num_clusters}


@app.post("/search")
def search(item: SearchInput):
    """
    Performs Nearest Neighbor Search (IVF).
    """
    # 1. Prepare Query
    query_np = np.array(item.query, dtype=np.float32)

    # 2. Run Search
    # Note: We use search_ivf. If index isn't built, it returns [].
    results = DB.search_ivf(query_np, item.k, item.nprobe)

    # 3. Check if we got results (if empty, maybe index wasn't built)
    if not results and DB.get_size() > 0:
        return {"warning": "No results found. Did you run POST /index?", "results": []}

    return {"results": results}


@app.post("/save")
def save_disk():
    """Manually triggers a save to disk."""
    DB.save(INDEX_FILE)
    return {"message": "Database saved to disk", "filename": INDEX_FILE}
