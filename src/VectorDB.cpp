// src/VectorDB.cpp

#include "VectorDB.hpp" // <--- We include the header we just made!

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