#include <pybind11/pybind11.h>
#include <pybind11/stl.h>   // Automatic conversion for std::vector, std::map, etc.
#include <pybind11/numpy.h> // The header required to handle NumPy arrays
#include "VectorDB.hpp"

// Create a namespace alias "py" to save typing "pybind11::" everywhere
namespace py = pybind11;

// =================================================================================
// THE MODULE ENTRY POINT
// =================================================================================
// PYBIND11_MODULE is a macro that creates the "init" function for Python.
// 1. "myvector_db": This MUST match the library name in CMakeLists.txt (and the .so filename).
// 2. "m": The variable representing the module instance.
// =================================================================================
PYBIND11_MODULE(myvector_db, m)
{

    // Module Documentation (appears in help(myvector_db))
    m.doc() = "A high-performance Vector Database plugin written in C++";

    // =============================================================================
    // BINDING THE CLASS
    // =============================================================================
    // py::class_<T> creates a Python class wrapper around the C++ type T.
    // Arguments:
    // 1. m: The module to attach this class to.
    // 2. "SimpleVectorDB": The name of the class in Python.
    // =============================================================================
    py::class_<SimpleVectorDB>(m, "SimpleVectorDB")

        // -------------------------------------------------------------------------
        // 1. The Constructor
        // Maps Python's __init__() to C++'s SimpleVectorDB()
        // -------------------------------------------------------------------------
        .def(py::init<>())

        // -------------------------------------------------------------------------
        // 2. Standard Methods
        // Maps Python method names to C++ member function addresses.
        // Syntax: .def("python_name", &Class::cpp_function_address)
        // -------------------------------------------------------------------------
        .def("add_vector", &SimpleVectorDB::add_vector)
        .def("get_size", &SimpleVectorDB::get_size)

        // -------------------------------------------------------------------------
        // 3. NumPy Integration (The "Adapter")
        // We cannot point to a standard C++ function here because C++ functions
        // don't speak "NumPy". We create a Lambda function to translate.
        // -------------------------------------------------------------------------
        .def("add_vector_numpy",
             [](SimpleVectorDB &self, py::array_t<float> input_array)
             {
                 // A. Request the "Buffer Info" struct from the Python object.
                 // This contains the raw pointer (ptr), size, and dimensions.
                 py::buffer_info buf = input_array.request();

                 // B. Input Validation
                 if (buf.ndim != 1)
                 {
                     throw std::runtime_error("Number of dimensions must be 1");
                 }

                 // C. Pointer Conversion
                 // buf.ptr is void* (blind). We cast it to float* so we can read it.
                 float *data_ptr = static_cast<float *>(buf.ptr);

                 // D. Size Extraction
                 size_t size = buf.size;

                 // E. Execution
                 // Call the raw C++ method using the extracted pointer and size.
                 self.add_vector_from_pointer(data_ptr, size);
             });
}