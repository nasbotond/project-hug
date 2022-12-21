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

/*
	The output of ICP algorithm
	trans : transformation for best align
	dictances[i] : the distance between node i in src and its nearst node in dst
	inter : number of iterations
*/
typedef struct
{
	Matrix4d trans;
	std::vector<float> distances;
	int iter;
} ICP_OUT;

typedef struct
{
    std::vector<float> distances;
    std::vector<int> indices;
} NEIGHBOR;


class ICP
{
    private:

    public:

        ICP(pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_in, pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_icp, int max_iter) : cloud_in(cloud_in), cloud_icp(cloud_icp), max_iter(max_iter) {}
        ~ICP() {}

        // Main functions
        Matrix4d best_fit_transform(const MatrixXd &A, const MatrixXd &B);
        ICP_OUT icp_alg(const MatrixXd &A, const MatrixXd &B, int max_iteration, float tolerance, int leaf_size=10, int Ksearch=5);
        void align(pcl::PointCloud<pcl::PointXYZ>& cloud_icp_);
        NEIGHBOR nearest_neighbor(const Eigen::MatrixXd &src, const Eigen::MatrixXd &dst);

        // Helper functions
        float dist(const Vector3d &a, const Vector3d &b);
        void setMaximumIterations(int iter);

        // Variables
        int max_iter;
        pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_in; // Original point cloud
        pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_icp; // ICP output point cloud
};