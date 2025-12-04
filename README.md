# VectorDB Project - Phase 1: The Foundations

This branch contains the "Toy Version" of a Vector Database. It demonstrates the fundamental bridge between high-performance C++ and Python using `pybind11`.

## **1. Core Concepts Learned**

### **C++ Memory & Structure**

-   **Header Files (`.hpp`):** The "Menu." Defines _what_ the class looks like (variables, function names).
-   **Source Files (`.cpp`):** The "Kitchen." Contains the actual logic/code.
-   **`std::vector`:** A dynamic array (like a Python List) that manages its own memory.
    -   `push_back(val)`: Adds an element to the end.
    -   `size()`: Returns the number of elements.
-   **Pass-by-Reference (`const std::vector<float>& vec`):**
    -   We use `&` to pass the _address_ of a vector to a function, avoiding a slow copy of the data.
    -   We use `const` to promise we won't modify the original data.

### **Project Structure**

```text
/vector_db_project
├── include/
│   └── VectorDB.hpp      // Class definition (The Interface)
├── src/
│   ├── VectorDB.cpp      // Class logic (The Implementation)
│   ├── bindings.cpp      // The Bridge (Pybind11 code)
│   └── main.cpp          // (Optional) Standalone C++ entry point
├── CMakeLists.txt        // Build instructions
└── test.py               // Python script to test the library
```

---

## **2. The Build System: CMake**

CMake is our "Project Manager." It generates a Makefile for us.

### **Key Rule**

-   **Space is King:** CMake arguments are separated by **spaces**, not commas.
    -   ✅ `add_executable(app src/main.cpp)`
    -   ❌ `add_executable(app, src/main.cpp)`

### **Scenario A: Building for `main.cpp` (Standalone Executable)**

Use this configuration if you want to run `./my_db` directly from the terminal without Python.

**`CMakeLists.txt` Snippet:**

```cmake
cmake_minimum_required(VERSION 3.10)
project(VectorDBProject)

set(CMAKE_CXX_STANDARD 17)
include_directories(include)

# Creates a standalone binary named "my_db"
add_executable(my_db
    src/main.cpp
    src/VectorDB.cpp
)
```

### **Scenario B: Building for `test.py` (Python Library)**

Use this configuration (current state) to `import myvector_db` in Python.

**`CMakeLists.txt` Snippet:**

```cmake
cmake_minimum_required(VERSION 3.10)
project(VectorDBProject)

set(CMAKE_CXX_STANDARD 17)
include_directories(include)

# 1. Download Pybind11
include(FetchContent)
FetchContent_Declare(
    pybind11
    GIT_REPOSITORY https://github.com/pybind/pybind11
    GIT_TAG        v2.11.1
)
FetchContent_MakeAvailable(pybind11)

# 2. Create the Python Module
# "myvector_db" is the name used in Python (import myvector_db)
pybind11_add_module(myvector_db
    src/VectorDB.cpp
    src/bindings.cpp
)
```

---

## **3. The Bridge: Pybind11**

Located in `src/bindings.cpp`. This file maps C++ concepts to Python.

-   **`#include <pybind11/stl.h>`**: The magic header. It automatically converts:
    -   Python `List` Wait \<---\> C++ `std::vector`
    -   Python `Dict` \<---\> C++ `std::map`
-   **`PYBIND11_MODULE`**: The macro that creates the module entry point.

---

## **4. How to Run**

### **Step 1: Build the C++ Library**

```bash
mkdir build
cd build
cmake ..   # Configure (Generate Makefile)
make       # Build (Generate .so file)
```

### **Step 2: Move the Library**

The build generates a file like `myvector_db.cpython-310-x86_64-linux-gnu.so`.
Copy this file to the root directory so Python can see it.

```bash
cp *.so ..
cd ..
```

### **Step 3: Run Python**

```bash
python3 test.py
```

## **5. Manual Compilation (The Hard Way)**

If you didn't have CMake, you would have to type these long commands manually. This is useful to understand what flags are required to build C++ code.

### **Scenario A: Standalone Executable**

Use this to build `main.cpp` into a runnable app called `my_db_app`.

```bash
g++ -std=c++17 -I./include src/main.cpp src/VectorDB.cpp -o my_db_app
```

-   `-std=c++17`: Force C++17 standard.
-   `-I./include`: Look for header files in the `include` folder.
-   `-o my_db_app`: Name the output file `my_db_app`.

### **Scenario B: Python Library**

_Prerequisite:_ For this command to work, you usually need to install pybind11 globally first: `pip install pybind11`.

```bash
g++ -O3 -Wall -shared -std=c++17 -fPIC \
    $(python3 -m pybind11 --includes) \
    -I./include \
    src/VectorDB.cpp src/bindings.cpp \
    -o myvector_db$(python3-config --extension-suffix)
```

**The Breakdown of the "Scary" Flags:**

-   `-O3`: Maximize optimization (make it fast\!).
-   `-Wall`: Show all warnings (helps catch bugs).
-   `-shared`: Create a **Shared Library** instead of an executable application.
-   `-fPIC`: **Position Independent Code**. Required for shared libraries so they can be loaded anywhere in memory.
-   `$(python3 -m pybind11 --includes)`: A shell command that asks Python: _"Where are the header files for Python and pybind11 located?"_
-   `$(python3-config --extension-suffix)`: A shell command that asks Python: _"What file extension should I use?"_ (It outputs something complex like `.cpython-310-x86_64-linux-gnu.so`).

### **Current Limitations (To be fixed in Phase 2)**

1.  **Double Copying:** `test.py` converts NumPy arrays to Python Lists, then C++ converts Lists to Vectors. This is slow.
2.  **Linear Search:** We are just storing data. We haven't implemented search algorithms yet.
3.  **No Persistence:** Data is lost when the script ends.
