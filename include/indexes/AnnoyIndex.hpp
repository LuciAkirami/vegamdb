// include/indexes/AnnoyIndex.hpp

#pragma once
#include "indexes/IndexBase.hpp"
#include <fstream>
#include <random>
#include <string>
#include <vector>

struct HyperPlane {
  std::vector<float> w;
  float bias = 0.0f;
};

struct AnnoyNode {
  HyperPlane *hyperplane;
  // float bias = 0.0f;
  std::vector<int> bucket;

  AnnoyNode *left = nullptr;
  AnnoyNode *right = nullptr;

  bool is_leaf() { return bucket.size() > 0; }

  ~AnnoyNode() {
    delete hyperplane;
    delete left;
    delete right;
  }

  AnnoyNode(int dimension) {
    hyperplane = new HyperPlane();
    hyperplane->w.resize(dimension);
  };
};

struct AnnoyIndexParams : SearchParams {
  int search_k_nodes;
};

class AnnoyIndex : public IndexBase {
private:
  int dimension;
  std::vector<AnnoyNode *> roots;
  int num_trees;
  int k_leaf;
  int search_k_nodes = 1;

public:
  AnnoyIndex(int dimension, int num_trees, int k_leaf, int search_k_nodes = 1);
  ~AnnoyIndex();
  virtual void build(const std::vector<std::vector<float>> &data) override;
  virtual SearchResults search(const std::vector<std::vector<float>> &data,
                               const std::vector<float> &query, int k,
                               const SearchParams *params = nullptr) override;
  virtual bool is_trained() const override;
  virtual void save(std::ofstream &out) const override;
  virtual void load(std::ifstream &in) override;
  virtual std::string name() const override { return "AnnoyIndex"; };

private:
  AnnoyNode *build_tree_recursive(const std::vector<std::vector<float>> &data,
                                  std::vector<int> &indices, std::mt19937 &rng);
  void free_tree(AnnoyNode *node);
  float get_margin(HyperPlane *hyperplane, const std::vector<float> &x);
  void create_hyperplane_for_split(const std::vector<std::vector<float>> &data,
                                   std::vector<int> &indices,
                                   HyperPlane *hyperplane, std::mt19937 &rng);
  void save_node(std::ofstream &out, AnnoyNode *node) const;
  AnnoyNode *load_node(std::ifstream &in);
};