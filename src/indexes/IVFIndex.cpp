// src/indexes//IVFIndex.cpp

#include "indexes/IVFIndex.hpp"
#include "indexes/IndexBase.hpp"
#include "indexes/KMeans.hpp"
#include "utils/Math.hpp"
#include <algorithm>
#include <cstddef>
#include <fstream>
#include <utility>
#include <vector>

IVFIndex::IVFIndex(int n_clusters, int dimension, int max_iters, int n_probe)
    : n_clusters(n_clusters), dimension(dimension), max_iters(max_iters),
      n_probe(n_probe) {}

SearchResults IVFIndex::search(const std::vector<std::vector<float>> &data,
                               const std::vector<float> &query, int k,
                               const SearchParams *params) {
  SearchResults results;

  size_t centroids_size = centroids.size();
  std::vector<std::pair<int, float>> centroid_scores;
  centroid_scores.resize(centroids_size);
  int effective_nprobe = this->n_probe; // Way 1: member default

  if (params) {
    auto ivf_params = dynamic_cast<const IVFSearchParams *>(params);
    if (ivf_params)
      effective_nprobe = ivf_params->n_probe; // Way 2 wins
  }

  int min_probe = std::min(effective_nprobe, static_cast<int>(centroids_size));

  for (int i = 0; i < centroids_size; i++) {
    float centroid_distance = euclidean_distance_squared(centroids[i], query);
    centroid_scores[i] = {i, centroid_distance};
  }

  std::sort(centroid_scores.begin(), centroid_scores.end(),
            [](const std::pair<int, float> &a, const std::pair<int, float> &b) {
              return a.second < b.second;
            });

  std::vector<std::pair<int, float>> candidate_scores;

  for (int i = 0; i < min_probe; i++) {
    int centroid_idx = centroid_scores[i].first;
    for (int j = 0; j < inverted_index[centroid_idx].size(); j++) {
      int vector_id = inverted_index[centroid_idx][j];
      float dist = euclidean_distance_squared(data[vector_id], query);
      candidate_scores.push_back({vector_id, dist});
    }
  }

  std::sort(candidate_scores.begin(), candidate_scores.end(),
            [](const std::pair<int, float> &a, const std::pair<int, float> &b) {
              return a.second < b.second;
            });

  int min_k = std::min(k, static_cast<int>(candidate_scores.size()));

  for (int i = 0; i < min_k; i++) {
    results.ids.push_back({candidate_scores[i].first});
    results.distances.push_back({candidate_scores[i].second});
  }

  return results;
}

void IVFIndex::build(const std::vector<std::vector<float>> &data) {
  KMeans kmeans_trainer(n_clusters, max_iters, dimension);

  KMeansIndex index = kmeans_trainer.train(data);

  centroids = index.centroids;
  inverted_index = index.buckets;
}

bool IVFIndex::is_trained() const {
  if (centroids.size() == 0 || inverted_index.size() == 0) {
    return false;
  }
  return true;
}

void IVFIndex::save(std::ofstream &out) const {
  if (!is_trained())
    return;

  int num_centroids = centroids.size();
  out.write(reinterpret_cast<const char *>(&num_centroids), sizeof(int));
  out.write(reinterpret_cast<const char *>(&dimension), sizeof(int));

  for (const auto &centroid : centroids) {
    out.write(reinterpret_cast<const char *>(centroid.data()),
              dimension * sizeof(float));
  }

  for (int i = 0; i < num_centroids; i++) {
    int bucket_size = inverted_index[i].size();

    out.write(reinterpret_cast<const char *>(&bucket_size), sizeof(int));
    out.write(reinterpret_cast<const char *>(inverted_index[i].data()),
              bucket_size * sizeof(int));
  }
}

void IVFIndex::load(std::ifstream &in) {
  in.read(reinterpret_cast<char *>(&n_clusters), sizeof(int));
  in.read(reinterpret_cast<char *>(&dimension), sizeof(int));

  centroids.resize(n_clusters);

  for (int i = 0; i < n_clusters; i++) {
    centroids[i].resize(dimension);
    in.read(reinterpret_cast<char *>(centroids[i].data()),
            dimension * sizeof(float));
  }

  inverted_index.resize(n_clusters);
  for (int i = 0; i < n_clusters; i++) {
    int bucket_size;
    in.read(reinterpret_cast<char *>(&bucket_size), sizeof(int));

    inverted_index[i].resize(bucket_size);
    in.read(reinterpret_cast<char *>(inverted_index[i].data()),
            bucket_size * sizeof(int));
  }
}