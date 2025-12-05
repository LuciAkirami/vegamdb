# 1. Check the CMake version
cmake_minimum_required(VERSION 3.10)

# 2. Name the project
project(VectorDBProject)

# 3. Set the C++ Standard
# We are forcing C++17. Why? 
# Because older C++ versions are painful. C++17 gives us modern features 
# (like better filesystem handling) that make life easier.
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

include(FetchContent)

FetchContent_Declare(
    pybind11
    GIT_REPOSITORY https://github.com/pybind/pybind11
    GIT_TAG        v2.11.1
)

FetchContent_MakeAvailable(pybind11)

# 4. Tell CMake where to look for "Menu" files (Headers)
# This is equivalent to the "-I./include" flag we typed manually.
include_directories(include)

# 5. Create the Executable
# "add_executable" tells CMake:
# "I want to build a program named 'my_db_app'.
#  Here are the ingredients (source files) you need to cook it."
# add_executable(my_db_app 
#     src/main.cpp 
#     src/VectorDB.cpp
# )

# 1. Turn on Maximum Optimization (-O3)
# 2. Tune for the architecture of YOUR specific computer (-march=native)
if(MSVC)
    add_compile_options(/O2 /arch:AVX2)
else()
    add_compile_options(-O3 -march=native -ffast-math)
endif()

pybind11_add_module(myvector_db
    src/VectorDB.cpp
    src/bindings.cpp
)