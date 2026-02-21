// src/bindings.cpp

#include "VegamDB.hpp"
#include "indexes/AnnoyIndex.hpp"
#include "indexes/FlatIndex.hpp"
#include "indexes/IVFIndex.hpp"
#include "indexes/IndexBase.hpp"
#include "indexes/KMeans.hpp"
#include <cstddef>
#include <pybind11/cast.h>
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

namespace py = pybind11;

PYBIND11_MODULE(_vegamdb, m) {

  m.doc() = "A high-performance Vector Database plugin written in C++";

  // ---- Return type ----
  py::class_<SearchResults>(m, "SearchResults",
                            R"(Container returned by VegamDB.search().

Attributes:
    ids (list[int]): Indices of the nearest neighbors in insertion order.
    distances (list[float]): Corresponding distances to the query vector.
)")
      .def_readonly("ids", &SearchResults::ids,
                    "List of integer indices of the nearest neighbors.")
      .def_readonly("distances", &SearchResults::distances,
                    "List of distances corresponding to each neighbor.");

  // ---- SearchParams hierarchy ----
  py::class_<SearchParams>(m, "SearchParams",
                           "Base class for index-specific search parameters.");

  py::class_<IVFSearchParams, SearchParams>(m, "IVFSearchParams",
                                            R"(Search parameters for IVF index.

Attributes:
    n_probe (int): Number of clusters to probe during search.
        Higher values improve recall at the cost of speed. Default: 1.

Example:
    params = IVFSearchParams()
    params.n_probe = 10
    results = db.search(query, k=5, params=params)
)")
      .def(py::init<>())
      .def_readwrite("n_probe", &IVFSearchParams::n_probe,
                     "Number of clusters to probe during search (default: 1).");

  // TODO: Uncomment once search_k_nodes runtime override is implemented
  // in AnnoyIndex::search()
  //
  // py::class_<AnnoyIndexParams, SearchParams>(
  //     m, "AnnoyIndexParams",
  //     R"(Search parameters for Annoy index.
  //
  // Attributes:
  //     search_k_nodes (int): Number of leaf nodes to inspect during search.
  //         Higher values improve recall at the cost of speed.
  //
  // Example:
  //     params = AnnoyIndexParams()
  //     params.search_k_nodes = 50
  //     results = db.search(query, k=5, params=params)
  // )")
  //     .def(py::init<>())
  //     .def_readwrite("search_k_nodes", &AnnoyIndexParams::search_k_nodes,
  //                    "Number of leaf nodes to inspect during search.");

  // ---- Index hierarchy ----
  py::class_<IndexBase>(m, "IndexBase",
                        "Abstract base class for all index types.");

  py::class_<FlatIndex, IndexBase>(
      m, "FlatIndex",
      "Brute-force flat index for exact nearest neighbor search.")
      .def(py::init<>());

  py::class_<IVFIndex, IndexBase>(
      m, "IVFIndex",
      "Inverted File Index using K-Means clustering for approximate search.")
      .def(py::init<int, int, int, int>(), py::arg("n_clusters"),
           py::arg("dimension"), py::arg("max_iters") = 50,
           py::arg("n_probe") = 1);

  py::class_<AnnoyIndex, IndexBase>(
      m, "AnnoyIndex",
      "Approximate Nearest Neighbors using random projection trees.")
      .def(py::init<int, int, int, int>(), py::arg("dimension"),
           py::arg("num_trees"), py::arg("k_leaf"),
           py::arg("search_k_nodes") = 1);

  // ---- VegamDB (the orchestrator) ----
  py::class_<VegamDB>(
      m, "VegamDB",
      "A high-performance vector database with pluggable index types.")
      .def(py::init<>(), "Create a new empty VegamDB instance.")
      .def("dimension", &VegamDB::dimension,
           "Return the dimensionality of stored vectors (0 if empty).")
      .def("add_vector", &VegamDB::add_vector, py::arg("vec"),
           "Add a single vector as a Python list of floats.")

      .def(
          "add_vector_numpy",
          [](VegamDB &self, py::array_t<float> input_array) {
            py::buffer_info buf = input_array.request();
            if (buf.ndim == 1) {
              float *data_ptr = static_cast<float *>(buf.ptr);
              size_t dim = buf.size;
              self.add_vector_np(data_ptr, 1, dim);
            } else if (buf.ndim == 2) {
              float *data_ptr = static_cast<float *>(buf.ptr);
              size_t n_vectors = buf.shape[0];
              size_t dim = buf.shape[1];
              self.add_vector_np(data_ptr, n_vectors, dim);
            } else {
              throw std::runtime_error("Number of dimensions must be 1/2D");
            }
          },
          py::arg("input_array"),
          "Add a single vector from a 1D NumPy float32 array or a 2D Numpy "
          "Array(zero-copy).")

      .def("size", &VegamDB::size,
           "Return the number of vectors stored in the database.")

      // Factory lambdas: pybind11 v2.11.1 can't directly bind functions taking
      // unique_ptr<AbstractBase> as a parameter. These lambdas construct the
      // index in C++ and call set_index() internally, avoiding the ownership
      // transfer issue.
      .def(
          "use_flat_index",
          [](VegamDB &self) { self.set_index(std::make_unique<FlatIndex>()); },
          "Set the index to brute-force flat search (exact, no training "
          "needed).")

      .def(
          "use_ivf_index",
          [](VegamDB &self, int n_clusters, int max_iters, int n_probe) {
            self.set_index(std::make_unique<IVFIndex>(
                n_clusters, self.dimension(), max_iters, n_probe));
          },
          py::arg("n_clusters"), py::arg("max_iters") = 50,
          py::arg("n_probe") = 1,
          R"(Set the index to IVF (Inverted File Index) for approximate search.

Args:
    n_clusters: Number of Voronoi cells (clusters) for partitioning.
    max_iters: Maximum K-Means iterations for training (default: 50).
    n_probe: Number of clusters to search at query time (default: 1).
)")

      .def(
          "use_annoy_index",
          [](VegamDB &self, int num_trees, int k_leaf, int search_k_nodes) {
            self.set_index(std::make_unique<AnnoyIndex>(
                self.dimension(), num_trees, k_leaf, search_k_nodes));
          },
          py::arg("num_trees"), py::arg("k_leaf"),
          py::arg("search_k_nodes") = 1,
          R"(Set the index to Annoy (Approximate Nearest Neighbors Oh Yeah).

Args:
    num_trees: Number of random projection trees to build.
    k_leaf: Maximum number of points in each leaf node.
    search_k_nodes: Number of leaf nodes to search at query time (default: 1).
)")

      .def("build_index", &VegamDB::build_index,
           "Explicitly build/train the current index on stored vectors.")
      .def("search", &VegamDB::search, py::arg("query"), py::arg("k"),
           py::arg("params") = nullptr,
           R"(Search for the k nearest neighbors of a query vector.

Args:
    query: 1D list of floats representing the query vector.
    k: Number of nearest neighbors to return.
    params: Optional IVFSearchParams or AnnoyIndexParams.

Returns:
    SearchResults with .ids (list[int]) and .distances (list[float]).
)")
      .def("save", &VegamDB::save, py::arg("filename"),
           "Save the database (vectors + index) to a binary file.")
      .def("load", &VegamDB::load, py::arg("filename"),
           "Load a database (vectors + index) from a binary file.");

  // ---- KMeans (standalone utility) ----
  py::class_<KMeansIndex>(m, "KMeansIndex",
                          "Result container for K-Means training.")
      .def_readonly("centroids", &KMeansIndex::centroids,
                    "List of cluster centroid vectors.")
      .def_readonly("buckets", &KMeansIndex::buckets,
                    "List of clusters, each containing vector indices.");

  py::class_<KMeans>(m, "KMeans", "Standalone K-Means clustering utility.")
      .def(py::init<int, int, int>(), py::arg("n_clusters"),
           py::arg("dimension"), py::arg("max_iters"),
           "Create a KMeans instance with given parameters.")
      .def("train", &KMeans::train, py::arg("data"),
           "Train K-Means on the provided data and return a KMeansIndex.");
}
