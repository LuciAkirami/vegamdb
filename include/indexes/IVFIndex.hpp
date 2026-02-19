// include/indexes/IVFIndex.hpp

#pragma once
#include "IndexBase.hpp"

struct IVFSearchParams : public SearchParams {
  int n_probe = 1;
};

class IVFIndex : public IndexBase {
private:
  // The Cluster Centers (K vectors)
  std::vector<std::vector<float>> centroids;

  // The Buckets (K lists of vector IDs)
  std::vector<std::vector<int>> inverted_index;

  // Number of Clusters to consider
  int n_probe;

  // Dimension of the vectors
  int dimension;

  // ----- KMeans Args -----
  // Number of clusters
  int n_clusters;

  // Number of iterations
  int max_iters;

public:
  IVFIndex(int n_clusters, int dimension, int max_iters = 50, int n_probe = 1);

  virtual void build(const std::vector<std::vector<float>> &data) override;
  virtual SearchResults search(const std::vector<std::vector<float>> &data,
                               const std::vector<float> &query, int k,
                               const SearchParams *params = nullptr) override;

  virtual bool is_trained() const override;
  virtual void save(std::ofstream &out) const override;
  virtual void load(std::ifstream &in) override;
  virtual std::string name() const override { return "IVFIndex"; };
};