// src/indexes/FlatIndex.cpp

#include "indexes/FlatIndex.hpp"
#include "indexes/IndexBase.hpp"
#include "utils/Math.hpp"
#include <algorithm>
#include <cstddef>
#include <utility>
#include <vector>

SearchResults FlatIndex::search(const std::vector<std::vector<float>> &data,
                                const std::vector<float> &query, int k,
                                const SearchParams *params) {
  SearchResults results;
  results.ids.reserve(k);
  results.distances.reserve(k);

  size_t size = data.size();

  std::vector<std::pair<int, float>> scores;
  scores.reserve(size);

  for (int i = 0; i < size; i++) {
    float distance = euclidean_distance_squared(data[i], query);
    scores.push_back({i, distance});
  }

  std::sort(scores.begin(), scores.end(),
            [](const std::pair<int, float> &a, const std::pair<int, float> &b) {
              return a.second < b.second;
            });

  int min_k = std::min(k, (int)scores.size());
  for (int i = 0; i < min_k; i++) {
    results.ids.push_back(scores[i].first);
    results.distances.push_back(scores[i].second);
  }

  return results;
}

void FlatIndex::build(const std::vector<std::vector<float>> &data) {
  // No-op: Flat search has no index to build
}
bool FlatIndex::is_trained() const {
  return true; // Always "ready" — no training needed
}
void FlatIndex::save(std::ofstream &out) const {
  // No-op: No index state to persist
}
void FlatIndex::load(std::ifstream &in) {
  // No-op: No index state to restore
}