// src/main.cpp

#include <iostream>
#include <vector>
#include "VectorDB.hpp" // Import our new database class

int main()
{
	// 1. Create an instance of our Database
	SimpleVectorDB my_db;

	// 2. Create some dummy data (embeddings)
	std::vector<float> vec1 = {0.1f, 0.2f, 0.3f};
	std::vector<float> vec2 = {0.4f, 0.5f, 0.6f};

	// 3. Add them to the DB
	std::cout << "Adding vectors..." << std::endl;
	my_db.add_vector(vec1);
	my_db.add_vector(vec2);

	// 4. Check if it worked
	std::cout << "Database size: " << my_db.get_size() << std::endl;

	return 0;
}