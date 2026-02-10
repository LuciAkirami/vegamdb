#pragma once
#include <vector>
#include <cmath>
#include <limits>
#include <random>

// =========================================================
// SECTION: Data Structures
// =========================================================

/**
 * @brief Holds the result of the K-Means training.
 * Contains the calculated centroids and the inverted index (buckets)
 * mapping each centroid to the list of vector indices assigned to it.
 */
struct KMeansIndex
{
    // The final positions of the K centroids
    std::vector<std::vector<float>> centroids;

    // The Inverted Index: Bucket[i] contains vector IDs belonging to Centroid[i]
    std::vector<std::vector<int>> buckets;
};

// =========================================================
// SECTION: Class Definition
// =========================================================

/**
 * @brief Implements Lloyd's Algorithm for K-Means Clustering.
 * Used to train the quantization layer for the IVF Index.
 */
class KMeans
{
private:
    int k;
    int max_iters;
    int dimension;

public:
    /**
     * @brief Constructor.
     * @param k Number of clusters (centroids) to find.
     * @param max_iters Maximum iterations for the training loop.
     * @param dimension Dimensionality of the vectors.
     */
    KMeans(int k, int max_iters, int dimension);

    /**
     * @brief Main Training Function.
     * Runs the clustering algorithm on the provided data.
     * @param data Reference to the dataset.
     * @return KMeansIndex Struct containing centroids and buckets.
     */
    KMeansIndex train(const std::vector<std::vector<float>> &data);

private:
    // =========================================================
    // SECTION: Internal Helpers
    // =========================================================

    /**
     * @brief Initialization Step.
     * Picks random points from the dataset to serve as initial centroids.
     */
    void init_centroids(const std::vector<std::vector<float>> &data, KMeansIndex &index);

    /**
     * @brief Assignment Step (Expectation).
     * Assigns every data point to its nearest centroid.
     */
    void assign_points_to_buckets(const std::vector<std::vector<float>> &data, KMeansIndex &index);

    /**
     * @brief Update Step (Maximization).
     * Recalculates centroid positions based on the average of their buckets.
     * Uses Row-Wise iteration for CPU cache optimization.
     */
    void update_centroids(const std::vector<std::vector<float>> &data, KMeansIndex &index);
};