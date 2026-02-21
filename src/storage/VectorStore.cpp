// src/storage/VectorStore.cpp

#include "storage/VectorStore.hpp"
#include <cstddef>
#include <fstream>
#include <vector>

void VectorStore::add(const std::vector<float> &vec) {
  if (data_.empty()) {
    this->dimension_ = vec.size();
  }

  this->data_.push_back(vec);
}

void VectorStore::add_vector_from_pointer(const float *arr, size_t n_vectors,
                                          size_t dim) {

  if (data_.empty()) {
    this->dimension_ = dim;
  }

  for (int i = 0; i < n_vectors; i++) {
    std::vector<float> temp(arr + i * dim, arr + i * dim + dim);

    this->data_.push_back(temp);
  }
}

const std::vector<float> &VectorStore::get(int idx) const {
  return this->data_[idx];
}

const std::vector<std::vector<float>> &VectorStore::data() const {
  return this->data_;
}

int VectorStore::size() const { return this->data_.size(); }

int VectorStore::dimension() const { return this->dimension_; }

void VectorStore::save(std::ofstream &out) const {
  int rows = this->data_.size(); // # vectors

  // Guard Clause: If DB is empty, don't create a file. Just return.
  if (rows == 0)
    return;

  int cols = this->data_[0].size(); // Dimensions

  out.write(reinterpret_cast<const char *>(&rows), sizeof(int));
  out.write(reinterpret_cast<const char *>(&cols), sizeof(int));

  for (const auto &vector : data_) {
    out.write(reinterpret_cast<const char *>(vector.data()),
              dimension_ * sizeof(float));
  }
}

void VectorStore::load(std::ifstream &in) {
  int rows, cols;
  in.read(reinterpret_cast<char *>(&rows), sizeof(int));
  in.read(reinterpret_cast<char *>(&cols), sizeof(int));

  this->dimension_ = cols;
  data_.resize(rows);

  for (int i = 0; i < rows; i++) {
    data_[i].resize(dimension_);
    in.read(reinterpret_cast<char *>(data_[i].data()),
            dimension_ * sizeof(float));
  }
}