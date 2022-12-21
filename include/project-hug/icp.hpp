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

/*
	The nearest neighbor point
	distance: the distence between src_matrix[sourceIndex] 
			  and dst_matrix[targetIndex]
*/
typedef struct
{
	int sourceIndex;
	int targetIndex;
	float distance;
} Align;

/*
	The k nearest neighbor points
	distances[i]: the distence between src_matrix[sourceIndex] 
			  	  and dst_matrix[targetIndexes[i]]
			      i := 1~K
*/
typedef struct
{
	int sourceIndex;
	std::vector<int> targetIndexes;
	std::vector<float> distances;
	float distanceMean;
} KNeighbor;


class ICP
{
    private:

    public:

        ICP(pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_in, pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_icp, int max_iter) : cloud_in(cloud_in), cloud_icp(cloud_icp), max_iter(max_iter) {}
        ~ICP() {}

        // Main functions
        Matrix4d best_fit_transform(const MatrixXd &A, const MatrixXd &B);
        Matrix4d best_fit_transform(const MatrixXd &A, const MatrixXd &B, std::vector<KNeighbor> neighbors, int remainPercentage=100, int K=5);
        std::vector<KNeighbor> k_nearest_neighbors(const MatrixXd& source, const MatrixXd& target, float leaf_size=10, int K=5);
        ICP_OUT icp_alg(const MatrixXd &A, const MatrixXd &B, int max_iteration, float tolerance, int leaf_size=10, int Ksearch=5);
        void align(pcl::PointCloud<pcl::PointXYZ>& cloud_icp_);

        // Helper functions
        float dist(const Vector3d &a, const Vector3d &b);
        void setMaximumIterations(int iter);
        // int cmpKNeighbor(const void *a, const void *b);

        // Variables
        int max_iter;
        pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_in; //(new pcl::PointCloud<pcl::PointXYZ>);  // Original point cloud
        pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_icp; //(new pcl::PointCloud<pcl::PointXYZ>);  // ICP output point cloud
};