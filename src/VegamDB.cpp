// src/VegamDB.cpp

#include "VegamDB.hpp"
#include "indexes/AnnoyIndex.hpp"
#include "indexes/FlatIndex.hpp"
#include "indexes/IVFIndex.hpp"
#include "indexes/IndexBase.hpp"
#include <cstddef>
#include <fstream>
#include <memory>

void VegamDB::add_vector(const std::vector<float> &vec) {
  this->store_.add(vec);
}

void VegamDB::add_vector_np(const float *arr, size_t n_vectors, size_t dim) {
  this->store_.add_vector_from_pointer(arr, n_vectors, dim);
}

int VegamDB::size() const { return this->store_.size(); }
int VegamDB::dimension() const { return this->store_.dimension(); }

void VegamDB::set_index(std::unique_ptr<IndexBase> index) {
  this->index_ = std::move(index);
}

void VegamDB::build_index() { this->index_->build(this->store_.data()); }

IndexBase *VegamDB::get_index() { return this->index_.get(); }

SearchResults VegamDB::search(const std::vector<float> &query, int k,
                              const SearchParams *params) {
  SearchResults results;
  if (this->index_) {

    if (this->index_->is_trained()) {
      results = this->index_->search(this->store_.data(), query, k, params);
      return results;
    }

    build_index();
    results = this->index_->search(this->store_.data(), query, k, params);
    return results;
  }

  auto flat_index = std::unique_ptr<IndexBase>(new FlatIndex());
  set_index(std::move(flat_index));
  build_index();

  results = this->index_->search(this->store_.data(), query, k, params);
  return results;
}

void VegamDB::save(const std::string &filename) {
  std::ofstream outfile(filename, std::ios::binary | std::ios::out);
  this->store_.save(outfile);

  // Write index type name (length-prefixed string)
  if (this->index_) {
    std::string index_name = this->index_->name();
    int name_len = index_name.size();
    outfile.write(reinterpret_cast<const char *>(&name_len), sizeof(int));
    outfile.write(index_name.data(), name_len);
    this->index_->save(outfile);
  }
}

void VegamDB::load(const std::string &filename) {
  std::ifstream infile(filename, std::ios::binary | std::ios::in);
  this->store_.load(infile);

  // Read index type name and construct the right index
  int name_len = 0;
  infile.read(reinterpret_cast<char *>(&name_len), sizeof(int));

  if (name_len > 0) {
    std::string index_name(name_len, '\0');
    infile.read(&index_name[0], name_len);
    int dim = this->store_.dimension();

    if (index_name == "IVFIndex") {
      // Construct with dummy params — load() will overwrite them
      this->index_ = std::make_unique<IVFIndex>(0, dim);
    } else if (index_name == "AnnoyIndex") {
      this->index_ = std::make_unique<AnnoyIndex>(dim, 0, 0);
    } else {
      this->index_ = std::make_unique<FlatIndex>();
    }

    this->index_->load(infile);
  }
}