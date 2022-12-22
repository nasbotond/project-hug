#pragma once

#include <iostream>
#include <filesystem>
#include <fstream>
#include <cstdlib>
#include <string>
#include <numeric>
#include <vector>
#include <Eigen/Eigen>
#include "nanoflann.hpp"
#include <pcl/io/ply_io.h>
#include <pcl/point_types.h>

using namespace Eigen;
using namespace nanoflann;

constexpr int SAMPLES_DIM = 3;

/*
	The output of ICP algorithm
	trans : transformation for best align
	dictances[i] : the distance between node i in src and its nearst node in dst
	inter : number of iterations
*/
typedef struct
{
	Matrix4d trans_mat;
	std::vector<float> distances;
	int iter;
} ICP_OUT;

typedef struct
{
    std::vector<float> distances;
    std::vector<int> A_indices;
    std::vector<int> B_indices;
} NEIGHBORS;

template <typename T, typename Compare>
std::vector<std::size_t> sort_permutation(const std::vector<T>& vec, Compare compare)
{
    std::vector<std::size_t> p(vec.size());
    std::iota(p.begin(), p.end(), 0);
    std::sort(p.begin(), p.end(), [&](std::size_t i, std::size_t j){ return compare(vec[i], vec[j]); });
    return p;
}

template <typename T>
std::vector<T> apply_permutation(const std::vector<T>& vec, const std::vector<std::size_t>& p)
{
    std::vector<T> sorted_vec(vec.size());
    std::transform(p.begin(), p.end(), sorted_vec.begin(),[&](std::size_t i){ return vec[i]; });
    return sorted_vec;
}

class ICP
{
    private:

    public:

        ICP(pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_in, pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_icp, int max_iter) : cloud_in(cloud_in), cloud_icp(cloud_icp), max_iter(max_iter) {}
        ~ICP() {}

        // Main functions
        Matrix4d best_fit_transform_SVD(const MatrixXd &A, const MatrixXd &B);
        Matrix4d best_fit_transform_quat(const MatrixXd &A, const MatrixXd &B);
        ICP_OUT icp_alg(const MatrixXd &A, const MatrixXd &B, int max_iteration, float tolerance, int leaf_size=10, int Ksearch=5);
        ICP_OUT tr_icp_alg(const MatrixXd &A, const MatrixXd &B, int max_iteration, float tolerance, int leaf_size=10, int Ksearch=5);
        void align(pcl::PointCloud<pcl::PointXYZ>& cloud_icp_);
        NEIGHBORS nearest_neighbor_naive(const Eigen::MatrixXd &src, const Eigen::MatrixXd &dst);
        NEIGHBORS nearest_neighbor_kdtree(const Eigen::MatrixXd &src, const Eigen::MatrixXd &dst);

        // Helper functions
        float dist(const Vector3d &a, const Vector3d &b); // for naive nearest neighbor
        void set_maximum_iterations(int iter);
        double trimmed_mse(const double &overlap, const std::vector<float> &distances);
        double get_overlap_parameter(const std::vector<float> &distances);

        // Variables
        int max_iter;
        pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_in; // Original point cloud
        pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_icp; // ICP output point cloud
};