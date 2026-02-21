// include/storage/VectorStore.hpp

#pragma once

#include <cstddef>
#include <fstream>
#include <vector>

class VectorStore {
private:
  std::vector<std::vector<float>> data_;
  int dimension_ = 0;

public:
  void add(const std::vector<float> &vec);
  void add_vector_from_pointer(const float *arr, size_t n_vectors, size_t dim);

  const std::vector<float> &get(int idx) const;
  const std::vector<std::vector<float>> &data() const;

  int size() const;
  int dimension() const;

  void save(std::ofstream &out) const;
  void load(std::ifstream &in);
};