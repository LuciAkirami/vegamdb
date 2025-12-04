// include/VectorDB.hpp

#pragma once
// ^ This is an "Include Guard".
// It prevents this file from being imported twice by accident,
// which would confuse the compiler.

#include <vector>

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

public:
	// "public" functions can be called by anyone using this class.

	// 1. Constructor: This runs automatically when you create the object.
	SimpleVectorDB();

	// 2. Function to add a vector to our database
	// "void" means it returns nothing.
	// "const std::vector<float>&" <- taking a read only reference of vector
	void add_vector(const std::vector<float> &vec);

	// 3. Function to get the current size of the database
	int get_size();
};