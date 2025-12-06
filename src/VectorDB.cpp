// src/VectorDB.cpp

#include "VectorDB.hpp" // <--- We include the header we just made!
#include <algorithm>
#include <cmath>
#include <fstream>
#include <stdexcept>

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

// 6. Save the database to a binary file
void SimpleVectorDB::save(const std::string &filename)
{
	int rows = database.size();

	// Guard Clause: If DB is empty, don't create a file. Just return.
	if (rows == 0)
	{
		return;
	}

	int cols = database[0].size();

	// Open file in Binary Mode.
	// std::ios::out is implied by ofstream, but good for clarity.
	std::ofstream outfile(filename, std::ios::binary | std::ios::out);

	if (!outfile.is_open())
	{
		throw std::runtime_error("Error: Could not open file for writing: " + filename);
	}

	// 1. Write Metadata (Header)
	// We cast the address of the integer (&rows) to a raw char pointer.
	// This dumps the 4 bytes of the integer directly to disk.
	outfile.write(reinterpret_cast<const char *>(&rows), sizeof(int));
	outfile.write(reinterpret_cast<const char *>(&cols), sizeof(int));

	// 2. Write Data (Body)
	// Iterate by Reference (const auto&) to avoid copying the vector.
	for (const auto &vec : database)
	{
		// vec.data() gives us the pointer to the raw array of floats.
		// We dump the entire row in one go.
		outfile.write(reinterpret_cast<const char *>(vec.data()), cols * sizeof(float));
	}

	outfile.close();
}

// 7. Load the database from a binary file (overwriting current data)
void SimpleVectorDB::load(const std::string &filename)
{
	std::ifstream infile(filename, std::ios::binary | std::ios::in);

	if (!infile.is_open())
	{
		throw std::runtime_error("Error: Could not open file for reading: " + filename);
	}

	int rows, cols;

	// 1. Read Metadata
	infile.read(reinterpret_cast<char *>(&rows), sizeof(int));
	infile.read(reinterpret_cast<char *>(&cols), sizeof(int));

	// 2. Prepare Memory
	// Clear ensures we remove old data. Resize allocates the "row" slots.
	database.clear();
	database.resize(rows);

	// 3. Read Data
	for (int i = 0; i < rows; i++)
	{
		// Allocate memory for this specific row so .read has a place to write.
		database[i].resize(cols);

		// Read directly from disk into the vector's internal memory array.
		// This avoids creating a temporary buffer.
		infile.read(reinterpret_cast<char *>(database[i].data()), cols * sizeof(float));
	}

	infile.close();
}