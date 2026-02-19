// include/indexes/FlatIndex.hpp

#pragma once
#include "IndexBase.hpp"
#include <fstream>
#include <string>
#include <vector>

// struct SearchResults {
//   std::vector<int> ids;
//   std::vector<float> distances;
// };

// struct SearchParams {
//   virtual ~SearchParams() = default;
// };

class FlatIndex : public IndexBase {
public:
  void build(const std::vector<std::vector<float>> &data) override;

  SearchResults search(const std::vector<std::vector<float>> &data,
                       const std::vector<float> &query, int k,
                       const SearchParams *params = nullptr) override;

  bool is_trained() const override;
  void save(std::ofstream &out) const override;
  void load(std::ifstream &in) override;
  std::string name() const override { return "FlatIndex"; };
};