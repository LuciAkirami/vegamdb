// src/utils/Math.cpp

#include "utils/Math.hpp"
#include <cmath>
#include <cstddef>
#include <random>
#include <vector>

float euclidean_distance(const std::vector<float> &a,
                         const std::vector<float> &b) {
  float sum = 0.0f;
  size_t size = a.size();

  for (size_t i = 0; i < size; i++) {
    float distance = a[i] - b[i];
    sum += distance * distance;
  }

  sum = std::sqrt(sum);

  return sum;
}

float euclidean_distance_squared(const std::vector<float> &a,
                                 const std::vector<float> &b) {
  float sum = 0.0f;
  size_t size = a.size();

  for (size_t i = 0; i < size; i++) {
    float distance = a[i] - b[i];
    sum += distance * distance;
  }

  return sum;
}

float dot_product(const std::vector<float> &a, const std::vector<float> &b) {
  float sum = 0.0f;
  size_t size = a.size();

  for (int i = 0; i < size; i++) {
    sum += a[i] * b[i];
  }

  return sum;
}

std::mt19937 get_random_engine() {
  std::random_device rd;
  return std::mt19937(rd());
}