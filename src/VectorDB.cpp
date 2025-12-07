// src/VectorDB.cpp

#include "VectorDB.hpp" // <--- We include the header we just made!
#include <algorithm>
#include <cmath>
#include <fstream>
#include <stdexcept>

// =========================================================
// Helper: Euclidean Distance
// Formula: d = sqrt( sum( (a[i] - b[i])^2 ) )
// Used heavily in both Brute Force search and KMeans.
// =========================================================
float SimpleVectorDB::dist(const std::vector<float> &a, const std::vector<float> &b)
{
	float distance = 0.0f;
	for (int i = 0; i < a.size(); i++)
	{
		float diff = a[i] - b[i];
		distance += (diff * diff);
	}
	return std::sqrt(distance);
}

// 1. Implementing the Constructor
// We use "SimpleVectorDB::" to tell the compiler:
// "This function belongs to the SimpleVectorDB class."
SimpleVectorDB::SimpleVectorDB()
{
	// Currently, we don't need to do anything special when starting.
	// The vector "database" is automatically created empty.
}

// 2. Implementing add_vector
void SimpleVectorDB::add_vector(const std::vector<float> &vec)
{
	// "push_back" is the standard way to add to a vector (like .append in Python)
	// We are adding the input "vec" into our big "database" vector.
	database.push_back(vec);
}

// 3. Implementing get_size
int SimpleVectorDB::get_size()
{
	// Return the number of vectors currently stored
	return database.size();
}

// 4. Implementing add_vector_from_pointer
void SimpleVectorDB::add_vector_from_pointer(const float *arr, size_t size)
{
	// 1. Create a "temporary" std::vector
	// We can initialize a vector directly from a pointer range!
	// This tells C++: "Start at address 'arr', and keep reading until you hit 'arr + size'"
	std::vector<float> vec(arr, arr + size);

	// 2. Add it to our database
	database.push_back(vec);
}

// 5. Implementing K Nearest Neighbors Search
std::vector<int> SimpleVectorDB::search(const std::vector<float> &input_query, int k)
{
	// 1. Store pairs of (Index, Distance)
	std::vector<std::pair<int, float>> scores;
	scores.reserve(database.size()); // Optimization: Reserve memory upfront to avoid resizing

	// 2. Calculate Distances
	for (int i = 0; i < database.size(); i++)
	{
		// CRITICAL: Use reference (&) to avoid copying the vector!
		const auto &target = database[i];

		float dist_sq = 0.0f;

		// 3. Euclidean Distance Calculation
		for (int j = 0; j < target.size(); j++)
		{
			float diff = target[j] - input_query[j];
			dist_sq += diff * diff;
		}

		// We push (Index, Distance)
		scores.push_back({i, std::sqrt(dist_sq)});
	}

	// 4. Sort based on Distance (Smallest distance first)
	std::sort(scores.begin(), scores.end(), [](const std::pair<int, float> &a, const std::pair<int, float> &b)
			  { return a.second < b.second; });

	// 5. Extract Top K Indices
	std::vector<int> indices;

	// Safety check: Don't try to return more items than exist in the DB!
	int top_k = std::min(k, (int)scores.size());

	for (int i = 0; i < top_k; i++)
	{
		// Push the Index (pair.first), NOT the distance
		indices.push_back(scores[i].first);
	}

	return indices;
}

// =========================================================
// Persistence: Save
// Serializes the Database AND the Index to a binary file.
// Format:
// [Rows][Cols] -> [Raw Vector Data] -> [HasIndex Flag] -> [Index Metadata] -> [Index Data]
// =========================================================
void SimpleVectorDB::save(const std::string &filename)
{
	int rows = database.size();

	// Guard Clause: If DB is empty, don't create a file. Just return.
	if (rows == 0)
		return;

	int cols = database[0].size();

	// Open file in Binary Mode.
	// std::ios::out is implied by ofstream, but good for clarity.
	std::ofstream outfile(filename, std::ios::binary | std::ios::out);

	if (!outfile.is_open())
	{
		throw std::runtime_error("Error: Could not open file for writing: " + filename);
	}

	// ---------------------------------------------------------
	// PART 1: Save Flat Data
	// ---------------------------------------------------------

	// Write Metadata (Header)
	// We cast the address of the integer (&rows) to a raw char pointer.
	// This dumps the 4 bytes of the integer directly to disk.
	// reinterpret_cast<const char*> tells C++: "Treat this address as just bytes, ignore types."
	outfile.write(reinterpret_cast<const char *>(&rows), sizeof(int));
	outfile.write(reinterpret_cast<const char *>(&cols), sizeof(int));

	// Write Data (Body)
	// Iterate by Reference (const auto&) to avoid copying the vector.
	for (const auto &vec : database)
	{
		// vec.data() gives us the pointer to the raw array of floats.
		// We dump the entire row in one go (cols * sizeof(float) bytes).
		outfile.write(reinterpret_cast<const char *>(vec.data()), cols * sizeof(float));
	}

	// ---------------------------------------------------------
	// PART 2: Save IVF Index
	// We use a flag to tell the loader if an index exists or not.
	// ---------------------------------------------------------
	int has_index = is_indexed ? 1 : 0;
	outfile.write(reinterpret_cast<const char *>(&has_index), sizeof(int));

	if (is_indexed)
	{
		// A. Write Index Dimensions
		// We need to know how many centroids (k) and their size (dim)
		int k = centroids.size();
		int dim = centroids[0].size();

		outfile.write(reinterpret_cast<const char *>(&k), sizeof(int));
		outfile.write(reinterpret_cast<const char *>(&dim), sizeof(int));

		// B. Write Centroids
		// Loop through each centroid and dump its raw floats
		for (const auto &centroid : centroids)
		{
			// BUG FIX REMINDER: Use 'centroid.data()', not 'centroids.data()'
			outfile.write(reinterpret_cast<const char *>(centroid.data()), dim * sizeof(float));
		}

		// C. Write Inverted Index (The Buckets)
		// Buckets are "Jagged Arrays" (variable lengths), so we cannot just dump them.
		// We must write [Size] -> [Data] for every bucket so load() knows how much to read.
		for (int i = 0; i < k; i++)
		{
			int bucket_size = inverted_index[i].size();

			// 1. Write size of this specific bucket
			outfile.write(reinterpret_cast<const char *>(&bucket_size), sizeof(int));

			// 2. Write the Vector IDs
			// Note: We are writing Integers here, not Floats
			outfile.write(reinterpret_cast<const char *>(inverted_index[i].data()), bucket_size * sizeof(int));
		}
	}

	outfile.close();
}

// =========================================================
// Persistence: Load
// Deserializes the binary file and reconstructs the Index in memory.
// =========================================================
void SimpleVectorDB::load(const std::string &filename)
{
	std::ifstream infile(filename, std::ios::binary | std::ios::in);

	if (!infile.is_open())
	{
		throw std::runtime_error("Error: Could not open file for reading: " + filename);
	}

	// ---------------------------------------------------------
	// PART 1: Load Flat Data
	// ---------------------------------------------------------
	int rows, cols;

	// Read Metadata
	infile.read(reinterpret_cast<char *>(&rows), sizeof(int));
	infile.read(reinterpret_cast<char *>(&cols), sizeof(int));

	// Prepare Memory
	// Clear ensures we remove old data. Resize allocates the "row" slots.
	database.clear();
	database.resize(rows);

	// Read Data
	for (int i = 0; i < rows; i++)
	{
		// Allocate memory for this specific row so .read has a place to write.
		// Without resize(cols), .data() would point to nothing!
		database[i].resize(cols);

		// Read directly from disk into the vector's internal memory array.
		// This avoids creating a temporary buffer.
		infile.read(reinterpret_cast<char *>(database[i].data()), cols * sizeof(float));
	}

	// ---------------------------------------------------------
	// PART 2: Load IVF Index
	// ---------------------------------------------------------
	int has_index;
	infile.read(reinterpret_cast<char *>(&has_index), sizeof(int));

	if (has_index == 1)
	{
		this->is_indexed = true;

		// A. Load Centroids
		int k, dim;
		infile.read(reinterpret_cast<char *>(&k), sizeof(int));
		infile.read(reinterpret_cast<char *>(&dim), sizeof(int));

		centroids.resize(k); // Allocates K empty vectors (slots)
		for (int i = 0; i < k; i++)
		{
			// BUG FIX REMINDER: Must resize the inner vector BEFORE reading into it!
			centroids[i].resize(dim);
			infile.read(reinterpret_cast<char *>(centroids[i].data()), dim * sizeof(float));
		}

		// B. Load Inverted Index
		inverted_index.resize(k);
		for (int i = 0; i < k; i++)
		{
			int bucket_size = 0;
			// Read how many items are in this bucket
			infile.read(reinterpret_cast<char *>(&bucket_size), sizeof(int));

			// Allocate memory for those items
			inverted_index[i].resize(bucket_size);

			// Read the Vector IDs
			infile.read(reinterpret_cast<char *>(inverted_index[i].data()), bucket_size * sizeof(int));
		}
	}
	else
	{
		this->is_indexed = false;
		// Good practice: clear any leftover index data from previous runs
		centroids.clear();
		inverted_index.clear();
	}

	infile.close();
}

// =========================================================
// Index Building (Training Phase)
// This function transforms the Flat Database into an IVF (Inverted File) Index.
//
// Concepts:
// 1. Centroids: The "center points" of the clusters.
// 2. Inverted Index: A list of buckets. Bucket[0] holds all Vector IDs belonging to Centroid[0].
// =========================================================
void SimpleVectorDB::build_index(int num_clusters, int max_iters)
{
	// 1. Guard Clause: Cannot train on empty data
	if (database.empty())
		return;

	int dim = database[0].size();

	// 2. Instantiate the Trainer
	// We delegate the hard math to the KMeans class
	KMeans trainer(num_clusters, max_iters, dim);

	// 3. Run Training (This takes time!)
	// Returns a struct containing {centroids, buckets}
	KMeansIndex results = trainer.train(database);

	// 4. Store Results
	// We save these into the SimpleVectorDB class so we can use them for search later.
	this->centroids = results.centroids;
	this->inverted_index = results.buckets;
	this->is_indexed = true; // Flag to enable search_ivf
}

// =========================================================
// IVF Search (The Optimized Search)
// Instead of scanning 1 Million vectors, we only scan a few relevant buckets.
//
// Arguments:
// - query: The vector we are looking for.
// - k: How many neighbors to return.
// - nprobe: Accuracy vs Speed knob. How many nearby clusters should we check?
//           (nprobe=1 is fastest, nprobe=100 is most accurate).
// =========================================================
std::vector<int> SimpleVectorDB::search_ivf(const std::vector<float> &query, int k, int nprobe)
{
	// 1. Safety Checks
	if (!is_indexed)
	{
		return {}; // Index not built yet!
	}

	// Clamp nprobe: Cannot check more clusters than actually exist.
	if (nprobe > this->centroids.size())
		nprobe = centroids.size();

	// ---------------------------------------------------------
	// STEP 1: Coarse Quantization (Find closest Clusters)
	// We compare the Query against the 100 Centroids to find where it "probably" lives.
	// ---------------------------------------------------------
	std::vector<std::pair<int, float>> centroid_dist_pairs;
	centroid_dist_pairs.reserve(centroids.size());

	for (int i = 0; i < centroids.size(); i++)
	{
		float d = dist(centroids[i], query);
		centroid_dist_pairs.push_back({i, d});
	}

	// Sort to find the 'nprobe' closest centroids
	std::sort(centroid_dist_pairs.begin(), centroid_dist_pairs.end(),
			  [](std::pair<int, float> &a, std::pair<int, float> &b)
			  { return a.second < b.second; }); // Return TRUE if a is smaller than b

	// ---------------------------------------------------------
	// STEP 2: Fine Search (Scan the Buckets)
	// We only look at vectors inside the top 'nprobe' buckets.
	// ---------------------------------------------------------
	std::vector<std::pair<int, float>> candidates;

	for (int i = 0; i < nprobe; i++)
	{
		// Get the ID of the ith closest cluster
		int cluster_id = centroid_dist_pairs[i].first;

		// Iterate through ONLY the vectors in that cluster
		for (int vector_id : inverted_index[cluster_id])
		{
			// Calculate exact distance to the actual vector
			float d = dist(database[vector_id], query);
			candidates.push_back({vector_id, d});
		}
	}

	// ---------------------------------------------------------
	// STEP 3: Selection (Top K)
	// Now we sort the candidates to find the actual nearest neighbors.
	// ---------------------------------------------------------
	std::sort(candidates.begin(), candidates.end(),
			  [](std::pair<int, float> &a, std::pair<int, float> &b)
			  { return a.second < b.second; });

	std::vector<int> final_indices;

	// Safety: Don't crash if we found fewer candidates than K
	int actual_k = std::min(k, (int)candidates.size());

	for (int i = 0; i < actual_k; i++)
	{
		final_indices.push_back(candidates[i].first);
	}

	return final_indices;
}