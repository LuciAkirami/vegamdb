#pragma once
#include <vector>
#include <string>
#include <fstream>
#include "KMeans.hpp"
#include "Utils.hpp"

// =========================================================
// SECTION: IVF (Inverted File) Index
// =========================================================

/**
 * @brief Inverted File Index.
 * Uses K-Means clustering to partition the search space.
 * * Logic:
 * 1. Train: Cluster data into K centroids (buckets).
 * 2. Search: Find nearest centroids, then search only those buckets.
 */
class IVFIndex
{
private:
    // The Cluster Centers (K vectors)
    std::vector<std::vector<float>> centroids;

    // The Buckets (K lists of vector IDs)
    std::vector<std::vector<int>> inverted_index;

    // Pointer to the raw database (We need this to calculate distances during search)
    const std::vector<std::vector<float>> *data_ptr = nullptr;

    // Dimension of the vectors
    int dimension = 0;

public:
    IVFIndex() = default;

    /**
     * @brief Builds the index using K-Means.
     * @param data Reference to the raw vector data.
     * @param num_clusters Number of centroids.
     * @param max_iters Training iterations.
     */
    void build(const std::vector<std::vector<float>> &data, int num_clusters, int max_iters);

    /**
     * @brief Approximate Nearest Neighbor Search.
     * @param query Query vector.
     * @param k Number of neighbors to find.
     * @param nprobe Number of buckets to inspect.
     * @return std::vector<int> Result indices.
     */
    std::vector<int> search(const std::vector<float> &query, int k, int nprobe);

    /**
     * @brief Saves the index metadata (Centroids + Buckets) to an open file stream.
     * Note: Does NOT save the raw vector data (VectorDB does that).
     */
    void save(std::ofstream &out);

    /**
     * @brief Loads the index metadata from an open file stream.
     */
    void load(std::ifstream &in);
};