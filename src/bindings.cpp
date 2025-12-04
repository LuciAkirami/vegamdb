#include <pybind11/pybind11.h>
#include <pybind11/stl.h> // <--- Teacher's Note: Crucial!
#include "VectorDB.hpp"

namespace py = pybind11;

// The Macro: PYBIND11_MODULE
// This defines the module name Python will use.
// "myvector_db" is the name you will type in Python: import myvector_db
// "m" is the variable representing the module itself.
PYBIND11_MODULE(myvector_db, m)
{

    m.doc() = "My efficient Vector Database plugin"; // Optional docstring

    // Binding the Class
    // py::class_<C++_Class_Name>(module, "Python_Class_Name")
    py::class_<SimpleVectorDB>(m, "SimpleVectorDB")
        .def(py::init<>())                              // Bind the constructor
        .def("add_vector", &SimpleVectorDB::add_vector) // Bind the function
        .def("get_size", &SimpleVectorDB::get_size);    // Bind the function
}