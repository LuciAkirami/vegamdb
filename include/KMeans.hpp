#pragma once
#include <vector>
#include <cmath>
#include <limits>
#include <random>

struct KMeansIndex
{
    std::vector<std::vector<float>> centroids;
    std::vector<std::vector<int>> buckets;
};

class KMeans
{
private:
    int k;
    int max_iters;
    int dimension;

public:
    KMeans(int k, int max_iters, int dimension);

    KMeansIndex train(const std::vector<std::vector<float>> &data);

private:
    // Helper 1: Pick random starting points
    void init_centroids(const std::vector<std::vector<float>> &data, KMeansIndex &index);

    // Helper 2: Loop through data and fill buckets
    void assign_points_to_buckets(const std::vector<std::vector<float>> &data, KMeansIndex &index);

    // Helper 3: Calculate new averages from buckets
    void update_centroids(const std::vector<std::vector<float>> &data, KMeansIndex &index);

    // Helpter 4: Calculate the Eucledian distance between two vectors
    float dist(const std::vector<float> &a, const std::vector<float> &b);
};