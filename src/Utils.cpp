#include "Utils.hpp"

// =========================================================
// SECTION: Math Implementations
// =========================================================

float euclidean_distance(const std::vector<float> &a, const std::vector<float> &b)
{
    return std::sqrt(euclidean_distance_squared(a, b));
}

float euclidean_distance_squared(const std::vector<float> &a, const std::vector<float> &b)
{
    float sum = 0.0f;
    size_t n = a.size();

    // Loop unrolling optimizations are handled by -O3 flag
    for (size_t i = 0; i < n; ++i)
    {
        float diff = a[i] - b[i];
        sum += diff * diff;
    }
    return sum;
}

float dot_product(const std::vector<float> &a, const std::vector<float> &b)
{
    float sum = 0.0f;
    size_t n = a.size();
    for (size_t i = 0; i < n; ++i)
    {
        sum += a[i] * b[i];
    }
    return sum;
}

// =========================================================
// SECTION: Random Implementations
// =========================================================

std::mt19937 get_random_engine()
{
    std::random_device rd;
    return std::mt19937(rd());
}
