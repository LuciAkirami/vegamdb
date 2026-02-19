// include/indexes/IndexBase.hpp

#pragma once
#include <fstream>
#include <string>
#include <vector>

struct SearchResults {
  std::vector<int> ids;
  std::vector<float> distances;
};

struct SearchParams {
  virtual ~SearchParams() = default;
};

class IndexBase {
public:
  virtual ~IndexBase() = default;

  virtual void build(const std::vector<std::vector<float>> &data) = 0;

  virtual SearchResults search(const std::vector<std::vector<float>> &data,
                               const std::vector<float> &query, int k,
                               const SearchParams *params = nullptr) = 0;

  virtual bool is_trained() const = 0;
  virtual void save(std::ofstream &out) const = 0;
  virtual void load(std::ifstream &in) = 0;
  virtual std::string name() const = 0;
};