import requests
import numpy as np
import time

BASE_URL = "http://127.0.0.1:8000"


def run_client_test():
    print("--- VECTOR DB CLIENT TEST ---")

    # 1. Health Check
    r = requests.get(f"{BASE_URL}/")
    print(f"Status: {r.json()}")

    # 2. Ingest Data (Add 1000 vectors via HTTP)
    print("\n[Client] Sending 1000 vectors via API...")
    dim = 128
    batch_size = 1000

    # Generate data locally
    data = np.random.random((batch_size, dim)).astype(np.float32)

    start = time.time()
    for i in range(batch_size):
        # Send JSON payload: {"vector": [0.1, 0.2...]}
        payload = {"vector": data[i].tolist()}
        requests.post(f"{BASE_URL}/vectors", json=payload)

    print(f"Ingestion took: {time.time() - start:.2f}s")

    # 3. Train Index
    print("\n[Client] Triggering Training...")
    payload = {"num_clusters": 10, "max_iters": 5}
    r = requests.post(f"{BASE_URL}/index", json=payload)
    print(f"Training response: {r.json()}")

    # 4. Search
    print("\n[Client] Searching...")
    query = np.random.random((dim)).tolist()
    search_payload = {"query": query, "k": 5, "nprobe": 5}

    start = time.time()
    r = requests.post(f"{BASE_URL}/search", json=search_payload)
    print(f"Search took: {time.time() - start:.4f}s")
    print(f"Results: {r.json()}")

    # 5. Save
    print("\n[Client] Saving...")
    requests.post(f"{BASE_URL}/save")
    print("Saved.")


if __name__ == "__main__":
    run_client_test()
