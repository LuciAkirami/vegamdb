// include/VegamDB.hpp

#pragma once

#include "indexes/IndexBase.hpp"
#include "storage/VectorStore.hpp"
#include <cstddef>
#include <memory>
class VegamDB {
private:
  VectorStore store_;
  std::unique_ptr<IndexBase> index_;

public:
  VegamDB() = default;

  // Data
  void add_vector(const std::vector<float> &vec);
  void add_vector_np(const float *arr, size_t size);
  int size() const;
  int dimension() const;

  // Index management
  void set_index(std::unique_ptr<IndexBase> index);
  void build_index();
  IndexBase *get_index();

  SearchResults search(const std::vector<float> &query, int k,
                       const SearchParams *params = nullptr);
  // Persistence
  void save(const std::string &filename);
  void load(const std::string &filename);
};