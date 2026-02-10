// include/VectorDB.hpp

#pragma once
// ^ This is an "Include Guard".
// It prevents this file from being imported twice by accident,
// which would confuse the compiler.

#include <vector>
#include <cstddef> // <--- Add this line! This defines "size_t"
#include <string>
#include "KMeans.hpp"
#include "Annoy.hpp"
#include "Utils.hpp" // <--- Include the new Utils module

// We define a Class. This is our "Blueprint".
class SimpleVectorDB
{

private:
	// "private" means only code INSIDE this class can touch this data.
	// Python doesn't really have this (everything is public).
	// In C++, we hide raw data to prevent bugs.

	// This is a vector of vectors! A 2D matrix.
	// Each row is one embedding (e.g., [0.1, 0.5, ...])
	std::vector<std::vector<float>> database;

	// ----- IVF Index Storage -----
	bool is_indexed = false;
	std::vector<std::vector<float>> centroids;
	std::vector<std::vector<int>> inverted_index;

	// ----- Annoy Index Storage -----
	AnnoyIndex *annoy_index = nullptr;

public:
	// "public" functions can be called by anyone using this class.

	// =========================================================
	// SECTION: Constructors & Basic Operations
	// =========================================================

	/**
	 * @brief Constructor: This runs automatically when you create the object.
	 */
	SimpleVectorDB();

	/**
	 * @brief Destructor: This runs automatically when the program exits.
	 */
	~SimpleVectorDB();

	/**
	 * @brief Function to add a vector to our database.
	 * "void" means it returns nothing.
	 * "const std::vector<float>&" <- taking a read only reference of vector
	 * @param vec The vector to add.
	 */
	void add_vector(const std::vector<float> &vec);

	/**
	 * @brief Function to get the current size of the database.
	 * @return int Number of vectors stored.
	 */
	int get_size();

	/**
	 * @brief Function to add a vector our database using raw pointer.
	 * This is used for Zero-Copy insertion from NumPy/Python.
	 * @param arr Pointer to the raw float array.
	 * @param size Number of elements in the array.
	 */
	void add_vector_from_pointer(const float *arr, size_t size);

	// =========================================================
	// SECTION: Standard Search (Brute Force)
	// =========================================================

	/**
	 * @brief Search Function (Brute Force).
	 * input_query: The vector to search for.
	 * k: How many neighbors to return (e.g., top 5).
	 * Returns: A list of indices (integers) of the nearest vectors.
	 * @param input_query The query vector.
	 * @param k Number of neighbors to return.
	 * @return std::vector<int> List of nearest neighbor indices.
	 */
	std::vector<int> search(const std::vector<float> &input_query, int k);

	// =========================================================
	// SECTION: Persistence (Save / Load)
	// =========================================================

	/**
	 * @brief Save the database to a binary file.
	 * Persists both the flat vector data and the IVF index (if built).
	 * @param filename Output filename.
	 */
	void save(const std::string &filename);

	/**
	 * @brief Load the database from a binary file(overwriting current data).
	 * Restores vectors and rebuilds the IVF index structures in memory.
	 * @param filename Input filename.
	 */
	void load(const std::string &filename);

	// =========================================================
	// SECTION: IVF (Inverted File) Indexing
	// =========================================================

	/**
	 * @brief Train the index.
	 * Uses K-Means clustering to partition the data into buckets.
	 * @param num_clusters Number of centroids to find.
	 * @param max_iters Max iterations for Lloyd's algorithm.
	 */
	void build_ivf_index(int num_clusters, int max_iters);

	/**
	 * @brief Fast Search (IVF).
	 * Uses the Inverted Index to scan only a subset of vectors.
	 * nprobe = How many nearby clusters to check (e.g. 1 or 3).
	 * @param query The query vector.
	 * @param k Number of neighbors.
	 * @param nprobe Number of buckets to inspect.
	 * @return std::vector<int> List of nearest neighbor indices.
	 */
	std::vector<int> search_ivf(const std::vector<float> &query, int k, int nprobe);

	// =========================================================
	// SECTION: Annoy (Tree-based) Indexing
	// =========================================================

	/**
	 * @brief Builds an Annoy Forest (Tree-based index).
	 * @param num_trees Number of trees to build (More = Better accuracy, Slower).
	 * @param k_leaf Max number of items in a leaf node.
	 */
	void build_annoy_index(int num_trees, int k_leaf = 100);

	/**
	 * @brief Searches using the Annoy Index.
	 * @param query The query vector.
	 * @param k Number of neighbors to return.
	 * @param search_k Backtracking limit (0 = Greedy search).
	 * @return std::vector<int> Indices of nearest neighbors.
	 */
	std::vector<int> search_annoy(const std::vector<float> &query, int k, int search_k = -1);
};