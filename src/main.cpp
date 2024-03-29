#include <iostream>
#include <string>
#include <random>

#include <pcl/io/ply_io.h>
#include <pcl/point_types.h>
#include <pcl/registration/icp.h>
#include <pcl/visualization/pcl_visualizer.h>
#include <pcl/console/time.h>   // TicToc

#include "icp.hpp"

typedef pcl::PointXYZ PointT;
typedef pcl::PointCloud<PointT> PointCloudT;

bool next_iteration = false;
void print4x4Matrix(const Eigen::Matrix4d & matrix)
{
    printf("Rotation matrix :\n");
    printf("    | %6.3f %6.3f %6.3f | \n", matrix (0, 0), matrix (0, 1), matrix (0, 2));
    printf("R = | %6.3f %6.3f %6.3f | \n", matrix (1, 0), matrix (1, 1), matrix (1, 2));
    printf("    | %6.3f %6.3f %6.3f | \n", matrix (2, 0), matrix (2, 1), matrix (2, 2));
    printf("Translation vector :\n");
    printf("t = < %6.3f, %6.3f, %6.3f >\n\n", matrix (0, 3), matrix (1, 3), matrix (2, 3));
}
void keyboardEventOccurred(const pcl::visualization::KeyboardEvent& event, void*)
{
    if(event.getKeySym() == "space" && event.keyDown ())
    {
        next_iteration = true;
    }        
}

int main(int argc, char* argv[])
{
    // The point clouds we will be using
    PointCloudT::Ptr cloud_in(new PointCloudT);  // Original point cloud
    PointCloudT::Ptr cloud_tr(new PointCloudT);  // Transformed point cloud
    PointCloudT::Ptr cloud_icp(new PointCloudT);  // ICP output point cloud

    // Checking program arguments
    if(argc < 2)
    {
        printf("Usage :\n");
        printf("\t\t%s model.ply data.ply <ICP version> <angle> <trans> <stddev> <mean> <number_of_ICP_iterations>\n", argv[0]);
        PCL_ERROR("Provide two ply files.\n");
        return(-1);
    }

    int version = 0;
    int angle = 8;
    double translation = 0.04;
    double stddev = 0.0;
    double mean = 0.0;
    int iterations = 1;  // Default number of ICP iterations
    if(argc > 3)
    {
        // If the user passed the version as an argument
        version = atoi(argv[3]);
        angle = atoi(argv[4]);
        translation = atof(argv[5]);
        stddev = atof(argv[6]);
        mean = atof(argv[7]);
        if(angle < 0)
        {
            printf("Angle has to be greater than 0!\n");
            return (-1);
        }
        if(version < 0 || version > 1)
        {
            printf("Version 0 for ICP and version 1 for TrICP!\n");
            return (-1);
        }
        if(argc > 8)
        {
            // If the user passed the number of iteration as an argument
            iterations = atoi(argv[8]);
            if(iterations < 1)
            {
                PCL_ERROR("Number of initial iterations must be >= 1\n");
                return (-1);
            }
        }
    }

    pcl::console::TicToc time;
    time.tic();
    if(pcl::io::loadPLYFile(argv[1], *cloud_in) < 0)
    {
        PCL_ERROR("Error loading cloud %s.\n", argv[1]);
        return (-1);
    }
    std::cout << "\nLoaded file " << argv[1] << " (" << cloud_in->size() << " points) in " << time.toc() << " ms\n" << std::endl;

    pcl::console::TicToc time1;
    time1.tic();
    if(pcl::io::loadPLYFile(argv[2], *cloud_icp) < 0)
    {
        PCL_ERROR("Error loading cloud %s.\n", argv[1]);
        return (-1);
    }
    std::cout << "\nLoaded file " << argv[2] << " (" << cloud_icp->size() << " points) in " << time1.toc() << " ms\n" << std::endl;

    // Defining a rotation matrix and translation vector
    Eigen::Matrix4d transformation_matrix = Eigen::Matrix4d::Identity();

    // A rotation matrix (see https://en.wikipedia.org/wiki/Rotation_matrix)
    // double theta = M_PI / 8;  // The angle of rotation in radians
    double theta = angle == 0 ? 0 : M_PI / angle;
    transformation_matrix(0, 0) = std::cos(theta);
    transformation_matrix(0, 1) = -sin(theta);
    transformation_matrix(1, 0) = sin(theta);
    transformation_matrix(1, 1) = std::cos(theta);

    // A translation on Z axis (0.4 meters)
    // transformation_matrix(2, 3) = 0.04;
    transformation_matrix(2, 3) = translation;
    // const double mean = 0.0;
    // const double stddev = 0.005;
    std::default_random_engine generator;
    std::normal_distribution<double> dist(mean, stddev);
    
    pcl::io::loadPLYFile(argv[2], *cloud_tr);

    // Display in terminal the transformation matrix
    std::cout << "Applying this rigid transformation to: cloud_icp -> cloud_icp" << std::endl;
    print4x4Matrix(transformation_matrix);

    pcl::transformPointCloud(*cloud_icp, *cloud_icp, transformation_matrix);
    pcl::transformPointCloud(*cloud_tr, *cloud_tr, transformation_matrix);

    // Add Gaussian noise          

    for(size_t point_i = 0; point_i < cloud_icp->points.size (); ++point_i)
    {
        cloud_icp->points[point_i].x += static_cast<float>(dist(generator));
        cloud_icp->points[point_i].y += static_cast<float>(dist(generator));
        cloud_icp->points[point_i].z += static_cast<float>(dist(generator));

        cloud_tr->points[point_i].x += static_cast<float>(dist(generator));
        cloud_tr->points[point_i].y += static_cast<float>(dist(generator));
        cloud_tr->points[point_i].z += static_cast<float>(dist(generator));
    }

    ICP icp = ICP(cloud_in, cloud_icp, iterations);
    // The Iterative Closest Point algorithm
    time.tic();
    // pcl::IterativeClosestPoint<PointT, PointT> icp;
    // icp.setMaximumIterations(iterations);
    // icp.setInputSource(cloud_icp);
    // icp.setInputTarget(cloud_in);
    // icp.align(*cloud_icp);
    // icp.setMaximumIterations(1);  // We set this variable to 1 for the next time we will call .align() function
    // PointCloudT::Ptr cloud_source_trans (new pcl::PointCloudT());
    Eigen::Matrix4d out_mat = icp.align(*cloud_icp, version);
    // std::cout << "Applied " << iterations << " ICP iteration(s) in " << time.toc() << " ms" << std::endl;

    print4x4Matrix(out_mat);

    Eigen::Matrix3d E = transformation_matrix.block<3,3>(0,0)*out_mat.block<3,3>(0,0); //.transpose();

    std::cout << "Rotation error (degrees): " << Eigen::AngleAxisd(E).angle()*(180/M_PI) << std::endl;
    std::cout << "Translation magnitude error: " << sqrt(transformation_matrix(2,3)*transformation_matrix(2,3)) - sqrt(out_mat(0,3)*out_mat(0,3) + out_mat(1,3)*out_mat(1,3) + out_mat(2,3)*out_mat(2,3)) << std::endl;

    icp.set_maximum_iterations(1);

    // Visualization
    pcl::visualization::PCLVisualizer viewer("ICP");
    // Create two vertically separated viewports
    int v1(0);
    int v2(1);
    viewer.createViewPort(0.0, 0.0, 0.5, 1.0, v1);
    viewer.createViewPort(0.5, 0.0, 1.0, 1.0, v2);

    // The color we will be using
    float bckgr_gray_level = 0.0;  // Black
    float txt_gray_lvl = 1.0 - bckgr_gray_level;

    // Original point cloud is white
    pcl::visualization::PointCloudColorHandlerCustom<PointT> cloud_in_color_h(cloud_in, (int) 255 * txt_gray_lvl, (int) 255 * txt_gray_lvl, (int) 255 * txt_gray_lvl);
    viewer.addPointCloud(cloud_in, cloud_in_color_h, "cloud_in_v1", v1);
    viewer.addPointCloud(cloud_in, cloud_in_color_h, "cloud_in_v2", v2);

    // Transformed point cloud is green
    pcl::visualization::PointCloudColorHandlerCustom<PointT> cloud_tr_color_h (cloud_tr, 20, 180, 20);
    viewer.addPointCloud(cloud_tr, cloud_tr_color_h, "cloud_tr_v1", v1);

    // ICP aligned point cloud is red
    pcl::visualization::PointCloudColorHandlerCustom<PointT> cloud_icp_color_h (cloud_icp, 180, 20, 20);
    viewer.addPointCloud(cloud_icp, cloud_icp_color_h, "cloud_icp_v2", v2);

    // Adding text descriptions in each viewport
    viewer.addText("White: Original point cloud\nGreen: Matrix transformed point cloud", 10, 15, 16, txt_gray_lvl, txt_gray_lvl, txt_gray_lvl, "icp_info_1", v1);
    viewer.addText("White: Original point cloud\nRed: ICP aligned point cloud", 10, 15, 16, txt_gray_lvl, txt_gray_lvl, txt_gray_lvl, "icp_info_2", v2);

    std::stringstream ss;
    ss << iterations;
    std::string iterations_cnt = "ICP iterations = " + ss.str ();
    viewer.addText(iterations_cnt, 10, 60, 16, txt_gray_lvl, txt_gray_lvl, txt_gray_lvl, "iterations_cnt", v2);

    // Set background color
    viewer.setBackgroundColor(bckgr_gray_level, bckgr_gray_level, bckgr_gray_level, v1);
    viewer.setBackgroundColor(bckgr_gray_level, bckgr_gray_level, bckgr_gray_level, v2);

    // Set camera position and orientation
    viewer.setCameraPosition(-3.68332, 2.94092, 5.71266, 0.289847, 0.921947, -0.256907, 0);
    viewer.setSize(1280, 1024);  // Visualiser window size

    // Register keyboard callback :
    viewer.registerKeyboardCallback(&keyboardEventOccurred, (void*) NULL);

    // Display the visualiser
    while(!viewer.wasStopped())
    {
        viewer.spinOnce();

        // The user pressed "space" :
        if(next_iteration)
        {
            // The Iterative Closest Point algorithm
            time.tic();
            icp.align(*cloud_icp, version);
            std::cout << "Applied 1 ICP iteration in " << time.toc() << " ms" << std::endl;
            ++iterations;

            ss.str("");
            ss << iterations;
            std::string iterations_cnt = "ICP iterations = " + ss.str();
            viewer.updateText(iterations_cnt, 10, 60, 16, txt_gray_lvl, txt_gray_lvl, txt_gray_lvl, "iterations_cnt");
            viewer.updatePointCloud(cloud_icp, cloud_icp_color_h, "cloud_icp_v2");
        }
        next_iteration = false;
    }
    return(0);
}