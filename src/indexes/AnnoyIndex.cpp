// src/indexes/AnnoyIndex.cpp

#include "indexes/AnnoyIndex.hpp"
#include "indexes/IndexBase.hpp"
#include "utils/Math.hpp"
#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <limits>
#include <queue>
#include <random>
#include <utility>
#include <vector>

AnnoyIndex::AnnoyIndex(int dimension, int num_trees, int k_leaf, int search_k,
                       bool use_priority_queue)
    : dimension(dimension), num_trees(num_trees), k_leaf(k_leaf),
      use_priority_queue(use_priority_queue) {
  if (search_k == -1) {
    this->search_k = num_trees * k_leaf;
  } else {
    this->search_k = search_k;
  }
};

AnnoyIndex::~AnnoyIndex() {
  for (auto *root : this->roots) {
    delete root;
  }
}

// void AnnoyIndex::free_tree(AnnoyNode *node) {
//   if (node == nullptr)
//     return;

//   free_tree(node->left);
//   free_tree(node->right);

//   delete node;
// }

void AnnoyIndex::create_hyperplane_for_split(
    const std::vector<std::vector<float>> &data, std::vector<int> &indices,
    HyperPlane *hyperplane, std::mt19937 &rng) {
  std::shuffle(indices.begin(), indices.end(), rng);
  int point_a_idx = indices[0];
  int point_b_idx = indices[1];

  std::vector<float> point_a = data[point_a_idx];
  std::vector<float> point_b = data[point_b_idx];

  // std::uniform_int_distribution<> dist(0, indices.size() - 1);
  // int db_idx_a, db_idx_b;

  // int rand_idx = dist(rng);
  // db_idx_a = indices[rand_idx];
  // db_idx_b = db_idx_a;

  // while (db_idx_a == db_idx_b) {
  //   rand_idx = dist(rng);
  //   db_idx_b = indices[rand_idx];
  // }

  // std::vector<float> point_a = data[db_idx_a];
  // std::vector<float> point_b = data[db_idx_b];

  for (int i = 0; i < dimension; i++) {
    float temp = point_a[i] - point_b[i];
    hyperplane->w[i] = temp;
    hyperplane->bias += temp * (point_a[i] + point_b[i]) / 2;
  }

  hyperplane->bias *= -1.0;
}

float AnnoyIndex::get_margin(HyperPlane *hyperplane,
                             const std::vector<float> &x) {
  // Substitute the point in the hyperplane equation
  float sum = 0;

  for (int i = 0; i < dimension; i++) {
    sum += x[i] * hyperplane->w[i];
  };

  sum += hyperplane->bias;

  return sum;
}

AnnoyNode *
AnnoyIndex::build_tree_recursive(const std::vector<std::vector<float>> &data,
                                 std::vector<int> &indices, std::mt19937 &rng) {

  AnnoyNode *node = new AnnoyNode(dimension);

  // Base case: small enough to be a leaf
  if (indices.size() <= k_leaf) {
    node->bucket = indices;
    return node;
  }

  std::vector<int> left_indices;
  std::vector<int> right_indices;

  create_hyperplane_for_split(data, indices, node->hyperplane, rng);

  for (int i = 0; i < indices.size(); i++) {
    const std::vector<float> &x = data[indices[i]];
    float margin = get_margin(node->hyperplane, x);

    if (margin > 0) {
      left_indices.push_back(indices[i]);
    } else {
      right_indices.push_back(indices[i]);
    }
  }

  if (left_indices.empty()) {
    node->bucket = right_indices;
    return node;
  }

  if (right_indices.empty()) {
    node->bucket = left_indices;
    return node;
  }

  node->left = build_tree_recursive(data, left_indices, rng);
  node->right = build_tree_recursive(data, right_indices, rng);

  return node;
}

void AnnoyIndex::build(const std::vector<std::vector<float>> &data) {
  // empty existing roots
  // if (!this->roots.empty()) {
  //   for (int i = 0; i < this->roots.size(); i++) {
  //     free_tree(this->roots[i]);
  //   }
  // }

  for (auto *root : roots) {
    delete root;
  }

  roots.clear();

  this->roots.resize(this->num_trees);

  for (int i = 0; i < num_trees; i++) {
    std::mt19937 rng = get_random_engine();
    std::vector<int> indices;
    indices.resize(data.size());

    std::iota(indices.begin(), indices.end(), 0);

    this->roots[i] = build_tree_recursive(data, indices, rng);
  }
}

SearchResults AnnoyIndex::search(const std::vector<std::vector<float>> &data,
                                 const std::vector<float> &query, int k,
                                 const SearchParams *params) {

  SearchResults results;

  if (!is_trained()) {
    return results;
  }

  int effective_search_k = this->search_k;
  bool effective_use_pq = this->use_priority_queue;

  if (params) {
    auto annoy_params = dynamic_cast<const AnnoyIndexParams *>(params);
    effective_search_k = annoy_params->search_k;
    effective_use_pq = annoy_params->use_priority_queue;
  }

  std::vector<int> candidates;

  if (effective_use_pq) {
    // --- Priority queue approach (Spotify-style) ---
    std::priority_queue<std::pair<float, AnnoyNode *>> pq;

    for (auto &root : this->roots) {
      // Cannot use infinity as we have enabled -O3
      pq.push({std::numeric_limits<float>::max(), root});
    }

    while ((candidates.size() < effective_search_k) && (!pq.empty())) {
      float distance = pq.top().first;
      AnnoyNode *node = pq.top().second;

      pq.pop();

      if (node->is_leaf()) {
        candidates.insert(candidates.end(), node->bucket.begin(),
                          node->bucket.end());

        continue;
      }

      float margin = get_margin(node->hyperplane, query);

      pq.push({std::min(distance, margin), node->left});
      pq.push({std::min(distance, -1.0f * margin), node->right});
    }
  } else {
    // --- Greedy approach (one leaf per tree) ---
    for (int i = 0; i < num_trees; i++) {
      AnnoyNode *curr = roots[i];

      while (!curr->is_leaf()) {
        float margin = get_margin(curr->hyperplane, query);

        if (margin >= 0.0) {
          curr = curr->left;
        } else {
          curr = curr->right;
        }
      }

      candidates.insert(candidates.end(), curr->bucket.begin(),
                        curr->bucket.end());
    }
  }

  std::sort(candidates.begin(), candidates.end());
  auto last = std::unique(candidates.begin(), candidates.end());
  candidates.erase(last, candidates.end());

  std::vector<std::pair<int, float>> candidate_scores;
  candidate_scores.resize(candidates.size());

  for (int i = 0; i < candidates.size(); i++) {
    int vector_idx = candidates[i];
    float distance = euclidean_distance_squared(query, data[vector_idx]);

    candidate_scores[i] = {vector_idx, distance};
  }

  std::sort(candidate_scores.begin(), candidate_scores.end(),
            [](const std::pair<int, float> &a, const std::pair<int, float> &b) {
              return a.second < b.second;
            });

  int min_k = std::min(k, static_cast<int>(candidate_scores.size()));

  for (int i = 0; i < min_k; i++) {
    results.ids.push_back(candidate_scores[i].first);
    results.distances.push_back(candidate_scores[i].second);
  }

  return results;
}

bool AnnoyIndex::is_trained() const { return !roots.empty(); }

void AnnoyIndex::save(std::ofstream &out) const {
  // Write metadata
  out.write(reinterpret_cast<const char *>(&use_priority_queue), sizeof(bool));
  out.write(reinterpret_cast<const char *>(&num_trees), sizeof(int));
  out.write(reinterpret_cast<const char *>(&dimension), sizeof(int));
  out.write(reinterpret_cast<const char *>(&k_leaf), sizeof(int));
  out.write(reinterpret_cast<const char *>(&search_k), sizeof(int));

  // Write each tree
  for (int i = 0; i < num_trees; i++) {
    save_node(out, roots[i]);
  }
}

void AnnoyIndex::save_node(std::ofstream &out, AnnoyNode *node) const {
  if (!node)
    return;

  bool leaf = node->is_leaf();
  out.write(reinterpret_cast<const char *>(&leaf), sizeof(bool));

  if (leaf) {
    int bucket_size = node->bucket.size();
    out.write(reinterpret_cast<const char *>(&bucket_size), sizeof(int));
    out.write(reinterpret_cast<const char *>(node->bucket.data()),
              bucket_size * sizeof(int));
  } else {
    // Write hyperplane
    out.write(reinterpret_cast<const char *>(node->hyperplane->w.data()),
              dimension * sizeof(float));
    out.write(reinterpret_cast<const char *>(&node->hyperplane->bias),
              sizeof(float));
    // Recurse — pre-order guarantees left is written before right
    save_node(out, node->left);
    save_node(out, node->right);
  }
}

void AnnoyIndex::load(std::ifstream &in) {
  in.read(reinterpret_cast<char *>(&use_priority_queue), sizeof(bool));
  in.read(reinterpret_cast<char *>(&num_trees), sizeof(int));
  in.read(reinterpret_cast<char *>(&dimension), sizeof(int));
  in.read(reinterpret_cast<char *>(&k_leaf), sizeof(int));
  in.read(reinterpret_cast<char *>(&search_k), sizeof(int));

  roots.resize(num_trees);
  for (int i = 0; i < num_trees; i++) {
    roots[i] = load_node(in);
  }
}

AnnoyNode *AnnoyIndex::load_node(std::ifstream &in) {
  AnnoyNode *node = new AnnoyNode(dimension);

  bool leaf;
  in.read(reinterpret_cast<char *>(&leaf), sizeof(bool));

  if (leaf) {
    int bucket_size;
    in.read(reinterpret_cast<char *>(&bucket_size), sizeof(int));
    node->bucket.resize(bucket_size);
    in.read(reinterpret_cast<char *>(node->bucket.data()),
            bucket_size * sizeof(int));
  } else {
    // Read hyperplane
    in.read(reinterpret_cast<char *>(node->hyperplane->w.data()),
            dimension * sizeof(float));
    in.read(reinterpret_cast<char *>(&node->hyperplane->bias), sizeof(float));
    // Recurse in same order as save
    node->left = load_node(in);
    node->right = load_node(in);
  }

  return node;
}
