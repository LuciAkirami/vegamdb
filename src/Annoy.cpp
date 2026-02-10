#include "Annoy.hpp"
#include <iostream>
#include <random>

AnnoyIndex::AnnoyIndex(int dim) : dimension(dim) {}

AnnoyIndex::~AnnoyIndex()
{
    for (AnnoyNode *root : roots)
    {
        free_tree(root);
    }
}

void AnnoyIndex::free_tree(AnnoyNode *node)
{
    if (node == nullptr)
        return;

    free_tree(node->left);
    free_tree(node->right);

    delete node;
}

float AnnoyIndex::get_margin(const std::vector<float> &w, const std::vector<float> &x, float bias)
{
    float dot = 0;
    for (int i = 0; i < dimension; i++)
    {
        dot += w[i] * x[i];
    }
    dot += bias;
    return dot;
}

void AnnoyIndex::create_split(const std::vector<int> &indices,
                              std::vector<float> &w,
                              float &bias,
                              std::mt19937 &rng)
{
    w.resize(dimension);

    std::uniform_int_distribution<> dist(0, indices.size() - 1);
    int db_idx_a, db_idx_b;

    int rand_idx = dist(rng);
    db_idx_a = indices[rand_idx];
    db_idx_b = db_idx_a;

    while (db_idx_a == db_idx_b)
    {
        rand_idx = dist(rng);
        db_idx_b = indices[rand_idx];
    }

    std::vector<float> vec_a = (*data_ptr)[db_idx_a];
    std::vector<float> vec_b = (*data_ptr)[db_idx_b];

    bias = 0.0f;
    for (int i = 0; i < dimension; i++)
    {
        w[i] = vec_a[i] - vec_b[i];
        bias += ((vec_a[i] - vec_b[i]) * (vec_a[i] + vec_b[i])) / 2.0;
    }
    bias = -1.0 * bias;
}

void AnnoyIndex::build(const std::vector<std::vector<float>> &data, int num_trees, int k_leaf)
{
    // this->data_ptr = &data;
    this->_data = data;
    this->data_ptr = &_data;
    std::random_device rd;
    std::mt19937 gen(rd());
    for (auto r : roots)
        free_tree(r);
    roots.clear();
    roots.resize(num_trees);

    for (int i = 0; i < num_trees; i++)
    {
        std::vector<int> all_indices(data.size());
        std::iota(all_indices.begin(), all_indices.end(), 0);
        roots[i] = build_tree_recursive(all_indices, k_leaf, gen);
    }
}

AnnoyNode *AnnoyIndex::build_tree_recursive(std::vector<int> &indices, int k_leaf, std::mt19937 &rng)
{
    AnnoyNode *node = new AnnoyNode();

    if (indices.size() <= k_leaf)
    {
        node->bucket = indices;
        return node;
    }

    create_split(indices, node->hyperplane, node->bias, rng);

    std::vector<int> left_indices;
    std::vector<int> right_indices;

    for (int i = 0; i < indices.size(); i++)
    {
        const std::vector<float> &v = (*data_ptr)[indices[i]];
        float margin = get_margin(node->hyperplane, v, node->bias);

        if (margin > 0)
        {
            left_indices.push_back(indices[i]);
        }
        else
        {
            right_indices.push_back(indices[i]);
        }
    }

    if (left_indices.size() == 0)
    {
        node->bucket = right_indices;
        return node;
    }

    if (right_indices.size() == 0)
    {
        node->bucket = left_indices;
        return node;
    }

    node->left = build_tree_recursive(left_indices, k_leaf, rng);
    node->right = build_tree_recursive(right_indices, k_leaf, rng);

    return node;
}

/**
 * @brief Calculates Euclidean distance between two vectors.
 * * @param a First vector.
 * @param b Second vector.
 * @return float The distance.
 */
float AnnoyIndex::dist(const std::vector<float> &a, const std::vector<float> &b)
{
    float distance = 0.0f;
    for (int i = 0; i < dimension; i++)
    {
        float diff = a[i] - b[i];
        distance += (diff * diff);
    }
    return distance;
}

std::vector<int> AnnoyIndex::query(const std::vector<float> &vec, int k, int search_k)
{
    std::vector<int> final_candidates;

    // 1. Gather Candidates
    for (auto root : roots)
    {
        AnnoyNode *curr = root; // Use a temp pointer to traverse
        while (!curr->is_leaf())
        {
            float margin = get_margin(curr->hyperplane, vec, curr->bias);
            if (margin > 0)
            {
                curr = curr->left;
            }
            else
            {
                curr = curr->right;
            }
        }
        // Optimization: insert is faster than loop push_back
        final_candidates.insert(final_candidates.end(), curr->bucket.begin(), curr->bucket.end());
    }

    // 2. Deduplicate
    std::sort(final_candidates.begin(), final_candidates.end());
    auto last = std::unique(final_candidates.begin(), final_candidates.end());
    final_candidates.erase(last, final_candidates.end());

    // 3. Score
    std::vector<std::pair<int, float>> scores;
    scores.reserve(final_candidates.size());

    for (int idx : final_candidates)
    {
        float d = dist((*data_ptr)[idx], vec);
        scores.push_back({idx, d});
    }

    // 4. Sort
    std::sort(scores.begin(), scores.end(), [](const std::pair<int, float> &a, const std::pair<int, float> &b)
              { return a.second < b.second; });

    // 5. Return Top K
    std::vector<int> final_list;

    // BUG FIX 2: Safety Check + Use 'k'
    int limit = std::min(k, (int)scores.size());

    for (int i = 0; i < limit; i++)
    {
        final_list.push_back(scores[i].first);
    }

    return final_list;
}