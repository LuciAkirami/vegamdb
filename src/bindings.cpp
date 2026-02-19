// src/bindings.cpp

#include "VegamDB.hpp"
#include "indexes/AnnoyIndex.hpp"
#include "indexes/FlatIndex.hpp"
#include "indexes/IVFIndex.hpp"
#include "indexes/IndexBase.hpp"
#include "indexes/KMeans.hpp"
#include <pybind11/cast.h>
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

namespace py = pybind11;

PYBIND11_MODULE(myvector_db, m) {

  m.doc() = "A high-performance Vector Database plugin written in C++";

  // ---- Return type ----
  py::class_<SearchResults>(m, "SearchResults")
      .def_readonly("ids", &SearchResults::ids)
      .def_readonly("distances", &SearchResults::distances);

  // ---- SearchParams hierarchy ----
  py::class_<SearchParams>(m, "SearchParams");

  py::class_<IVFSearchParams, SearchParams>(m, "IVFSearchParams")
      .def(py::init<>())
      .def_readwrite("n_probe", &IVFSearchParams::n_probe);

  py::class_<AnnoyIndexParams, SearchParams>(m, "AnnoyIndexParams")
      .def(py::init<>())
      .def_readwrite("search_k_nodes", &AnnoyIndexParams::search_k_nodes);

  // ---- Index hierarchy ----
  py::class_<IndexBase>(m, "IndexBase");

  py::class_<FlatIndex, IndexBase>(m, "FlatIndex").def(py::init<>());

  py::class_<IVFIndex, IndexBase>(m, "IVFIndex")
      .def(py::init<int, int, int, int>(), py::arg("n_clusters"),
           py::arg("dimension"), py::arg("max_iters") = 50,
           py::arg("n_probe") = 1);

  py::class_<AnnoyIndex, IndexBase>(m, "AnnoyIndex")
      .def(py::init<int, int, int, int>(), py::arg("dimension"),
           py::arg("num_trees"), py::arg("k_leaf"),
           py::arg("search_k_nodes") = 1);

  // ---- VegamDB (the orchestrator) ----
  py::class_<VegamDB>(m, "VegamDB")
      .def(py::init<>())
      .def("dimension", &VegamDB::dimension)
      .def("add_vector", &VegamDB::add_vector)

      .def("add_vector_numpy",
           [](VegamDB &self, py::array_t<float> input_array) {
             py::buffer_info buf = input_array.request();
             if (buf.ndim != 1) {
               throw std::runtime_error("Number of dimensions must be 1");
             }
             float *data_ptr = static_cast<float *>(buf.ptr);
             size_t size = buf.size;
             self.add_vector_np(data_ptr, size);
           })

      .def("size", &VegamDB::size)

      //   .def("set_index", &VegamDB::set_index)
      // Factory lambdas: pybind11 v2.11.1 can't directly bind functions taking
      // unique_ptr<AbstractBase> as a parameter. These lambdas construct the
      // index in C++ and call set_index() internally, avoiding the ownership
      // transfer issue.
      .def("use_flat_index",
           [](VegamDB &self) { self.set_index(std::make_unique<FlatIndex>()); })

      .def(
          "use_ivf_index",
          [](VegamDB &self, int n_clusters, int max_iters, int n_probe) {
            self.set_index(std::make_unique<IVFIndex>(
                n_clusters, self.dimension(), max_iters, n_probe));
          },
          py::arg("n_clusters"), py::arg("max_iters") = 50,
          py::arg("n_probe") = 1)

      .def(
          "use_annoy_index",
          [](VegamDB &self, int num_trees, int k_leaf, int search_k_nodes) {
            self.set_index(std::make_unique<AnnoyIndex>(
                self.dimension(), num_trees, k_leaf, search_k_nodes));
          },
          py::arg("num_trees"), py::arg("k_leaf"),
          py::arg("search_k_nodes") = 1)

      .def("build_index", &VegamDB::build_index)
      .def("search", &VegamDB::search, py::arg("query"), py::arg("k"),
           py::arg("params") = nullptr)
      .def("save", &VegamDB::save)
      .def("load", &VegamDB::load);

  // ---- KMeans (standalone utility) ----
  py::class_<KMeansIndex>(m, "KMeansIndex")
      .def_readonly("centroids", &KMeansIndex::centroids)
      .def_readonly("buckets", &KMeansIndex::buckets);

  py::class_<KMeans>(m, "KMeans")
      .def(py::init<int, int, int>())
      .def("train", &KMeans::train);
}
