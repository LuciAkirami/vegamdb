#pragma once
#include <vector>
#include <cmath>
#include <random>
#include <algorithm>

// 1. The Node Structure
struct AnnoyNode
{
    // --- Inner Node Data (The Split) ---
    // The "Normal Vector" perpendicular to the split plane
    std::vector<float> hyperplane;
    // The offset from origin
    float bias = 0.0f;

    // --- Leaf Node Data (The Bucket) ---
    // If this node is a leaf, it stores the IDs of the vectors that landed here.
    std::vector<int> bucket;

    // --- Structure ---
    AnnoyNode *left = nullptr;
    AnnoyNode *right = nullptr;

    // Helper to check type
    bool is_leaf() const
    {
        return bucket.size() > 0;
    }
};

// 2. The Index Class
class AnnoyIndex
{
private:
    int dimension;

    // We create a "Forest" (Multiple trees).
    // This vector stores the Root Node of each tree.
    std::vector<AnnoyNode *> roots;

    // We need access to the actual vector data to calculate splits.
    // We use a pointer to avoid copying the huge DB.
    const std::vector<std::vector<float>> *data_ptr = nullptr;
    std::vector<std::vector<float>> _data;

public:
    AnnoyIndex(int dim);
    ~AnnoyIndex(); // Destructor to clean up memory!

    // Main API
    // num_trees: How many trees to build (More trees = Better accuracy, Slower build)
    // k_leaf: Max items in a leaf node before we stop splitting
    void build(const std::vector<std::vector<float>> &data, int num_trees, int k_leaf);

    // Search API
    // query: The vector to search for
    // k: How many neighbors to return
    // search_k: How many nodes to inspect (Accuracy knob)
    std::vector<int> query(const std::vector<float> &vec, int k, int search_k);

private:
    // Recursive Builder
    // indices: The subset of vector IDs we are currently trying to split
    AnnoyNode *build_tree_recursive(std::vector<int> &indices, int k_leaf, std::mt19937 &rng);

    // Recursive Helper to delete nodes
    void free_tree(AnnoyNode *node);

    // Math Helpers
    float get_margin(const std::vector<float> &w, const std::vector<float> &x, float bias);
    void create_split(const std::vector<int> &indices, std::vector<float> &w, float &bias, std::mt19937 &rng);
    float dist(const std::vector<float> &a, const std::vector<float> &b);
};