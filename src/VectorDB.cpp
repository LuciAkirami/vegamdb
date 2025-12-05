// src/VectorDB.cpp

#include "VectorDB.hpp" // <--- We include the header we just made!
#include <algorithm>
#include <cmath>

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