// include/utils/Math.hpp

#pragma once
#include <cmath>
#include <random>
#include <vector>

/**
 * @brief Calculates the Standard Euclidean Distance (L2 Norm).
 * Formula: sqrt(sum((a - b)^2))
 * @param a The first vector.
 * @param b The second vector.
 * @return The distance between a and b.
 */
float euclidean_distance(const std::vector<float> &a,
                         const std::vector<float> &b);

/**
 * @brief Calculates Squared Euclidean Distance.
 * Formula: sum((a - b)^2)
 * * Use this function for sorting or finding the "closest" item,
 * as it avoids the computationally expensive sqrt() operation
 * while preserving relative order.
 * @param a The first vector.
 * @param b The second vector.
 * @return The squared distance.
 */
float euclidean_distance_squared(const std::vector<float> &a,
                                 const std::vector<float> &b);

/**
 * @brief Calculates the Dot Product of two vectors.
 * Formula: sum(a[i] * b[i])
 * - Used primarily for Hyperplane calculations in Annoy.
 * @param a The first vector.
 * @param b The second vector.
 * @return The dot product result.
 */
float dot_product(const std::vector<float> &a, const std::vector<float> &b);

/**
 * @brief Returns a seeded Mersenne Twister random engine.
 * * Initializes a std::random_device once to seed the engine.
 * * Use this to avoid creating and seeding generators inside loops.
 * @return std::mt19937 A ready-to-use random number generator.
 */
std::mt19937 get_random_engine();