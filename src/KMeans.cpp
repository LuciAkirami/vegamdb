#include "KMeans.hpp"
#include <vector>
#include <cmath>
#include <algorithm>
#include <numeric>

KMeans::KMeans(int k, int max_iters, int dimension) : k(k), max_iters(max_iters), dimension(dimension) {}

// =========================================================
// Helper: Euclidean Distance
// Calculates distance between two vectors: sqrt(sum((a-b)^2))
// =========================================================
float KMeans::dist(const std::vector<float> &a, const std::vector<float> &b)
{
    float distance = 0.0f;
    for (int i = 0; i < dimension; i++)
    {
        float diff = a[i] - b[i];
        distance += (diff * diff);
    }
    return std::sqrt(distance);
}

/* ---- Single Train Function ---------
// KMeansIndex KMeans::train(std::vector<std::vector<float>> &data)
// {
//     if (k > data.size())
//         return;

//     std::random_device rd;
//     std::mt19937 gen(rd());

//     std::vector<int> indices(data.size());
//     std::iota(indices.begin(), indices.end(), 0);

//     std::shuffle(indices.begin(), indices.end(), gen);

//     std::vector<std::vector<float>> centroids;
//     centroids.reserve(k);

//     for (int i = 0; i < k; i++)
//     {
//         centroids.push_back(data[indices[i]]);
//     }

//     int tries = 0;

//     std::vector<std::vector<int>> bucket;
//     bucket.resize(k);

//     while (tries < max_iters)
//     {
//         for (int i = 0; i < data.size(); i++)
//         {
//             std::vector<std::pair<int, float>> score;
//             for (int j = 0; j < k; j++)
//             {
//                 float distance = dist(data[i], centroids[j]);
//                 score.push_back({j, distance});
//             }
//             std::sort(score.begin(), score.end(), [](std::pair<int, float> &a, std::pair<int, float> &b)
//                       { a.second < b.second; });

//             int centroidIndex = score[0].first;
//             bucket[centroidIndex].push_back(i);
//         }

//         for (int x = 0; x < bucket.size(); x++)
//         {
//             std::vector<float> centroid;
//             centroid.resize(dimension);

//             for (int y = 0; y < dimension; y++)
//             {
//                 float sum = 0;
//                 for (int z = 0; z < bucket[x].size(); z++)
//                 {
//                     sum += data[bucket[x][z]][y];
//                 }
//                 centroid[y] = (sum / bucket[x].size());
//             }

//             centroids[x] = centroid;
//         }

//         tries++;
//     }

//     struct KMeansIndex index;
//     index.centroids = centroids;
//     index.buckets = bucket;
// }
*/

// =========================================================
// Main Function: Train
// Implements Lloyd's Algorithm:
// 1. Initialize Centroids
// 2. Assignment Step (Points -> Nearest Centroid)
// 3. Update Step (Centroid -> Average of Points)
// 4. Repeat until max_iters
// =========================================================
KMeansIndex KMeans::train(const std::vector<std::vector<float>> &data)
{
    KMeansIndex index;

    // Safety Check: Cannot find K clusters if we have fewer than K data points
    if (k > data.size())
        return index;

    // 1. Setup Memory
    index.centroids.resize(k);
    index.buckets.resize(k);

    // 2. Initialize Starting Positions
    // We pick K random points from the data to be the starting centroids
    init_centroids(data, index);

    // 3. The Training Loop
    for (int iter = 0; iter < max_iters; iter++)
    {
        // Step A: Reset Buckets
        // We need empty buckets for the new round of assignments
        for (auto &b : index.buckets)
            b.clear();

        // Step B: Assignment Phase
        // Loop through all data points and assign them to the closest centroid
        assign_points_to_buckets(data, index);

        // Step C: Update Phase
        // Move centroids to the mathematical center (mean) of their buckets
        update_centroids(data, index);
    }

    return index;
}

// =========================================================
// Helper 1: Initialization
// strategy: Random Partitioning (Shuffle indices and pick K Centroids)
// =========================================================
void KMeans::init_centroids(const std::vector<std::vector<float>> &data, KMeansIndex &index)
{
    std::random_device rd;  // Hardware source of entropy
    std::mt19937 gen(rd()); // Mersenne Twister engine

    // Create a list of indices [0, 1, 2, ... N]
    std::vector<int> indices(data.size());
    std::iota(indices.begin(), indices.end(), 0);

    // Shuffle them randomly
    std::shuffle(indices.begin(), indices.end(), gen);

    // Pick the first K indices as our initial centroids
    for (int i = 0; i < k; i++)
    {
        index.centroids[i] = data[indices[i]];
    }
}

// =========================================================
// Helper 2: Assignment
// Finds the nearest centroid for every data point
// Time Complexity: O(N * K * Dimension)
// =========================================================
void KMeans::assign_points_to_buckets(const std::vector<std::vector<float>> &data, KMeansIndex &index)
{
    // Iterate through every vector in the dataset
    for (int i = 0; i < data.size(); i++)
    {
        int best_centroid_index = -1;
        float min_dist = std::numeric_limits<float>::max(); // Start with Infinity

        // Compare against all K centroids to find the closest one
        for (int j = 0; j < k; j++)
        {
            float d = dist(data[i], index.centroids[j]);
            if (d < min_dist)
            {
                min_dist = d;
                best_centroid_index = j;
            }
        }

        // Record the assignment
        // "Vector i belongs to Cluster j"
        index.buckets[best_centroid_index].push_back(i);
    }
}

// =========================================================
// Helper 3: Update
// Calculates the new mean position for each centroid.
// Uses Row-Wise Access for CPU Cache Optimization.
// =========================================================
void KMeans::update_centroids(const std::vector<std::vector<float>> &data, KMeansIndex &index)
{
    for (int i = 0; i < k; i++)
    {
        // Edge Case: If a cluster is empty (no points assigned), skip update.
        // Doing otherwise would cause a divide-by-zero crash.
        if (index.buckets[i].empty())
            continue;

        // 1. Summation Loop
        // We create a temporary accumulator vector
        std::vector<float> new_center(dimension, 0.0f);

        // OPTIMIZATION: Row-Wise Access
        // Instead of looping Dimensions first, we loop Vectors first.
        // This ensures we read contiguous blocks of memory (cache friendly).
        for (int vector_id : index.buckets[i])
        {
            for (int d = 0; d < dimension; d++)
            {
                new_center[d] += data[vector_id][d];
            }
        }

        // 2. Division Loop
        // Divide by the count to get the Average (Mean)
        float count = static_cast<float>(index.buckets[i].size());
        for (int d = 0; d < dimension; d++)
        {
            new_center[d] /= count;
        }

        // 3. Update the official centroid position
        index.centroids[i] = new_center;
    }
}