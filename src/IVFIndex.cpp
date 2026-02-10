#include "IVFIndex.hpp"
#include "Utils.hpp"
#include <algorithm>
#include <iostream>

void IVFIndex::build(const std::vector<std::vector<float>> &data, int num_clusters, int max_iters)
{
    if (data.empty())
        return;

    // FIX 1: Initialize dimension!
    this->dimension = data[0].size();
    this->data_ptr = &data;

    KMeans trainer(num_clusters, max_iters, dimension);
    KMeansIndex results = trainer.train(data);

    this->centroids = results.centroids;
    this->inverted_index = results.buckets;
}

std::vector<int> IVFIndex::search(const std::vector<float> &query, int k, int nprobe)
{
    if (this->centroids.empty())
        return {};

    // 1. Coarse Search (Find best clusters)
    std::vector<std::pair<int, float>> centroid_scores;
    centroid_scores.reserve(centroids.size());

    for (int i = 0; i < centroids.size(); i++)
    {
        // Euclidean squared is fine for sorting
        float d = euclidean_distance_squared(query, centroids[i]);
        // FIX 2: Use push_back, not index access on reserved vector
        centroid_scores.push_back({i, d});
    }

    std::sort(centroid_scores.begin(), centroid_scores.end(),
              [](const std::pair<int, float> &a, const std::pair<int, float> &b)
              { return a.second < b.second; });

    // FIX 3: Simplify clamping logic
    if (nprobe > centroids.size())
        nprobe = centroids.size();

    // 2. Fine Search (Scan buckets)
    std::vector<std::pair<int, float>> candidates;

    for (int i = 0; i < nprobe; i++)
    {
        int cluster_id = centroid_scores[i].first;

        // FIX 4: Correct loop variable (j++) and logic
        for (int vector_id : inverted_index[cluster_id])
        {
            // Access data via pointer
            float d = euclidean_distance_squared(query, (*data_ptr)[vector_id]);
            // FIX 5: Push back to candidates list
            candidates.push_back({vector_id, d});
        }
    }

    // 3. Selection
    std::sort(candidates.begin(), candidates.end(),
              [](const std::pair<int, float> &a, const std::pair<int, float> &b)
              { return a.second < b.second; });

    std::vector<int> final_list;

    // FIX 6: Simple clamp for K
    int actual_k = std::min(k, (int)candidates.size());

    for (int i = 0; i < actual_k; i++)
    {
        final_list.push_back(candidates[i].first);
    }

    return final_list;
}