/* 
Developer: Chunran Zheng <zhengcr@connect.hku.hk>

This file is subject to the terms and conditions outlined in the 'LICENSE' file,
which is included as part of this source code package.
*/

#ifndef LIDAR_DETECT_HPP
#define LIDAR_DETECT_HPP
#include <sensor_msgs/msg/point_cloud2.hpp>
#include <geometry_msgs/msg/point_stamped.hpp>
#include <Eigen/Dense>
#include <opencv2/opencv.hpp>
#include <rclcpp/rclcpp.hpp>
#include <pcl/filters/voxel_grid.h>
#include <pcl/filters/passthrough.h>
#include <pcl/point_types.h>
#include <pcl/io/pcd_io.h>
#include "common_lib.h"

class LidarDetect
{
private:
    double x_min_, x_max_, y_min_, y_max_, z_min_, z_max_;
    double circle_radius_;
    std::shared_ptr<rclcpp::Node> node_; // 添加节点引用用于日志

    // 存储中间结果的点云
    pcl::PointCloud<pcl::PointXYZ>::Ptr filtered_cloud_;
    pcl::PointCloud<pcl::PointXYZ>::Ptr plane_cloud_;
    pcl::PointCloud<pcl::PointXYZ>::Ptr aligned_cloud_;
    pcl::PointCloud<pcl::PointXYZ>::Ptr edge_cloud_;
    pcl::PointCloud<pcl::PointXYZ>::Ptr center_z0_cloud_;

public:
    rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr filtered_pub_;
    rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr plane_pub_;
    rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr aligned_pub_;
    rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr edge_pub_;
    rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr center_z0_pub_;
    rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr center_pub_;

    LidarDetect(std::shared_ptr<rclcpp::Node> node, Params &params)
        : node_(node),
          filtered_cloud_(new pcl::PointCloud<pcl::PointXYZ>),
          plane_cloud_(new pcl::PointCloud<pcl::PointXYZ>),
          aligned_cloud_(new pcl::PointCloud<pcl::PointXYZ>),
          edge_cloud_(new pcl::PointCloud<pcl::PointXYZ>),
          center_z0_cloud_(new pcl::PointCloud<pcl::PointXYZ>)
    {
        x_min_ = params.x_min;
        x_max_ = params.x_max;
        y_min_ = params.y_min;
        y_max_ = params.y_max;
        z_min_ = params.z_min;
        z_max_ = params.z_max;
        circle_radius_ = params.circle_radius;

        filtered_pub_ = node->create_publisher<sensor_msgs::msg::PointCloud2>("filtered_cloud", 1);
        plane_pub_ = node->create_publisher<sensor_msgs::msg::PointCloud2>("plane_cloud", 1);
        aligned_pub_ = node->create_publisher<sensor_msgs::msg::PointCloud2>("aligned_cloud", 1);
        edge_pub_ = node->create_publisher<sensor_msgs::msg::PointCloud2>("edge_cloud", 1);
        center_z0_pub_ = node->create_publisher<sensor_msgs::msg::PointCloud2>("center_z0_cloud", 10);
        center_pub_ = node->create_publisher<sensor_msgs::msg::PointCloud2>("center_cloud", 10);
    }

    void detect_lidar(pcl::PointCloud<pcl::PointXYZ>::Ptr cloud, pcl::PointCloud<pcl::PointXYZ>::Ptr center_cloud)
    {
        // 1. X、Y、Z方向滤波
        filtered_cloud_->reserve(cloud->size());

        pcl::PassThrough<pcl::PointXYZ> pass_x;
        pass_x.setInputCloud(cloud);
        pass_x.setFilterFieldName("x");
        pass_x.setFilterLimits(x_min_, x_max_);  // 设置X轴范围
        pass_x.filter(*filtered_cloud_);
    
        pcl::PassThrough<pcl::PointXYZ> pass_y;
        pass_y.setInputCloud(filtered_cloud_);
        pass_y.setFilterFieldName("y");
        pass_y.setFilterLimits(y_min_, y_max_);  // 设置Y轴范围
        pass_y.filter(*filtered_cloud_);
    
        pcl::PassThrough<pcl::PointXYZ> pass_z;
        pass_z.setInputCloud(filtered_cloud_);
        pass_z.setFilterFieldName("z");
        pass_z.setFilterLimits(z_min_, z_max_);  // 设置Z轴范围
        pass_z.filter(*filtered_cloud_);
    
        // 替换ROS_INFO为ROS2日志
        RCLCPP_INFO(node_->get_logger(), "Filtered cloud size: %ld", filtered_cloud_->size());
        
        pcl::VoxelGrid<pcl::PointXYZ> voxel_filter;
        voxel_filter.setInputCloud(filtered_cloud_);
        voxel_filter.setLeafSize(0.005f, 0.005f, 0.005f);
        voxel_filter.filter(*filtered_cloud_);
        RCLCPP_INFO(node_->get_logger(), "Filtered cloud size: %ld", filtered_cloud_->size());

        // 2. 平面分割
        plane_cloud_->reserve(filtered_cloud_->size());

        pcl::ModelCoefficients::Ptr plane_coefficients(new pcl::ModelCoefficients);
        pcl::PointIndices::Ptr plane_inliers(new pcl::PointIndices);
        pcl::SACSegmentation<pcl::PointXYZ> plane_segmentation;
        plane_segmentation.setModelType(pcl::SACMODEL_PLANE);
        plane_segmentation.setMethodType(pcl::SAC_RANSAC);
        plane_segmentation.setDistanceThreshold(0.05);  // 平面分割阈值
        plane_segmentation.setInputCloud(filtered_cloud_);
        plane_segmentation.segment(*plane_inliers, *plane_coefficients);
    
        pcl::ExtractIndices<pcl::PointXYZ> extract;
        extract.setInputCloud(filtered_cloud_);
        extract.setIndices(plane_inliers);
        extract.filter(*plane_cloud_);
        RCLCPP_INFO(node_->get_logger(), "Plane cloud size: %ld", plane_cloud_->size());
    
        // 3. 平面点云对齐   
        aligned_cloud_->reserve(plane_cloud_->size());

        Eigen::Vector3d normal(plane_coefficients->values[0],
            plane_coefficients->values[1],
            plane_coefficients->values[2]);
        normal.normalize();
        Eigen::Vector3d z_axis(0, 0, 1);

        Eigen::Vector3d axis = normal.cross(z_axis);
        double angle = acos(normal.dot(z_axis));

        Eigen::AngleAxisd rotation(angle, axis);
        Eigen::Matrix3d R = rotation.toRotationMatrix();

        // 应用旋转矩阵，将平面对齐到 Z=0 平面
        float average_z = 0.0;
        int cnt = 0;
        for (const auto& pt : *plane_cloud_) {
            Eigen::Vector3d point(pt.x, pt.y, pt.z);
            Eigen::Vector3d aligned_point = R * point;
            aligned_cloud_->push_back(pcl::PointXYZ(aligned_point.x(), aligned_point.y(), 0.0));
            average_z += aligned_point.z();
            cnt++;
        }
        average_z /= cnt;

        // 4. 提取边缘点
        edge_cloud_->reserve(aligned_cloud_->size());

        pcl::NormalEstimation<pcl::PointXYZ, pcl::Normal> normal_estimator;
        pcl::PointCloud<pcl::Normal>::Ptr normals(new pcl::PointCloud<pcl::Normal>);
        normal_estimator.setInputCloud(aligned_cloud_);
        normal_estimator.setRadiusSearch(0.03); // 设置法线估计的搜索半径
        normal_estimator.compute(*normals);
    
        pcl::PointCloud<pcl::Boundary> boundaries;
        pcl::BoundaryEstimation<pcl::PointXYZ, pcl::Normal, pcl::Boundary> boundary_estimator;
        boundary_estimator.setInputCloud(aligned_cloud_);
        boundary_estimator.setInputNormals(normals);
        boundary_estimator.setRadiusSearch(0.03); // 设置边界检测的搜索半径
        boundary_estimator.setAngleThreshold(M_PI / 4); // 设置角度阈值
        boundary_estimator.compute(boundaries);
    
        for (size_t i = 0; i < aligned_cloud_->size(); ++i) {
            if (boundaries.points[i].boundary_point > 0) {
                edge_cloud_->push_back(aligned_cloud_->points[i]);
            }
        }
        RCLCPP_INFO(node_->get_logger(), "Extracted %ld edge points.", edge_cloud_->size());

        // 5. 对边缘点进行聚类
        pcl::search::KdTree<pcl::PointXYZ>::Ptr tree(new pcl::search::KdTree<pcl::PointXYZ>);
        tree->setInputCloud(edge_cloud_);
    
        std::vector<pcl::PointIndices> cluster_indices;
        pcl::EuclideanClusterExtraction<pcl::PointXYZ> ec;
        ec.setClusterTolerance(0.02); // 设置聚类距离阈值
        ec.setMinClusterSize(50);     // 最小点数
        ec.setMaxClusterSize(1000);   // 最大点数
        ec.setSearchMethod(tree);
        ec.setInputCloud(edge_cloud_);
        ec.extract(cluster_indices);
    
        pcl::PointCloud<pcl::PointXYZRGB>::Ptr colored_cloud(new pcl::PointCloud<pcl::PointXYZRGB>);
        pcl::PointCloud<pcl::PointXYZRGB>::Ptr colored_clusters(new pcl::PointCloud<pcl::PointXYZRGB>);
        
        // 将原始边缘点云转换为彩色点云（灰色）
        for (const auto& point : edge_cloud_->points) {
            pcl::PointXYZRGB colored_point;
            colored_point.x = point.x;
            colored_point.y = point.y;
            colored_point.z = point.z;
            colored_point.r = 128;
            colored_point.g = 128;
            colored_point.b = 128;
            colored_cloud->push_back(colored_point);
        }
        
        // 定义颜色数组
        std::vector<std::array<uint8_t, 3>> colors = {
            {255, 0, 0},    // 红色 - Cluster 0
            {0, 255, 0},    // 绿色 - Cluster 1
            {0, 0, 255},    // 蓝色 - Cluster 2
            {255, 255, 0},  // 黄色 - Cluster 3
            {255, 0, 255},  // 紫色 - Cluster 4
            {0, 255, 255},  // 青色 - Cluster 5
            {255, 128, 0},  // 橙色 - Cluster 6
            {128, 255, 0}   // 浅绿色 - Cluster 7
        };
        RCLCPP_INFO(node_->get_logger(), "Number of edge clusters: %ld", cluster_indices.size());
    
        // 6. 对每个聚类进行圆拟合
        center_z0_cloud_->reserve(4);
        Eigen::Matrix3d R_inv = R.inverse();
    
        // 对每个聚类进行圆拟合
        for (size_t i = 0; i < cluster_indices.size(); ++i) 
        {
            pcl::PointCloud<pcl::PointXYZ>::Ptr cluster(new pcl::PointCloud<pcl::PointXYZ>);
            for (const auto& idx : cluster_indices[i].indices) {
                cluster->push_back(edge_cloud_->points[idx]);
            }
    
            RCLCPP_INFO(node_->get_logger(), "Cluster %ld: %ld points", i, cluster->size());
            
            // 为每个聚类添加彩色点
            auto color = colors[i % colors.size()];
            for (const auto& idx : cluster_indices[i].indices) {
                pcl::PointXYZRGB colored_point;
                colored_point.x = edge_cloud_->points[idx].x;
                colored_point.y = edge_cloud_->points[idx].y;
                colored_point.z = edge_cloud_->points[idx].z;
                colored_point.r = color[0];
                colored_point.g = color[1];
                colored_point.b = color[2];
                colored_clusters->push_back(colored_point);
            }
    
            // 圆拟合
            pcl::ModelCoefficients::Ptr coefficients(new pcl::ModelCoefficients);
            pcl::PointIndices::Ptr inliers(new pcl::PointIndices);
            pcl::SACSegmentation<pcl::PointXYZ> seg;
            seg.setOptimizeCoefficients(true);
            seg.setModelType(pcl::SACMODEL_CIRCLE2D);
            seg.setMethodType(pcl::SAC_RANSAC);
            seg.setDistanceThreshold(0.01); // 设置距离阈值
            seg.setMaxIterations(1000);     // 设置最大迭代次数
            seg.setInputCloud(cluster);
            seg.segment(*inliers, *coefficients);
    
            if (inliers->indices.size() > 0) 
            {
                double fitted_radius = coefficients->values[2];
                RCLCPP_INFO(node_->get_logger(), "Cluster %ld: fitted circle center (%.3f, %.3f), radius %.3f, inliers %ld", 
                           i, coefficients->values[0], coefficients->values[1], fitted_radius, inliers->indices.size());
                
                // 计算拟合误差
                double error = 0.0;
                for (const auto& idx : inliers->indices) 
                {
                    double dx = cluster->points[idx].x - coefficients->values[0];
                    double dy = cluster->points[idx].y - coefficients->values[1];
                    double distance = sqrt(dx * dx + dy * dy) - circle_radius_; // 距离误差
                    error += abs(distance);
                }
                error /= inliers->indices.size();
    
                RCLCPP_INFO(node_->get_logger(), "Cluster %ld: fitting error %.4f (threshold 0.025), expected radius %.3f", 
                           i, error, circle_radius_);
    
                // 如果拟合误差较小，则认为是一个圆洞
                if (error < 0.025) 
                {
                    RCLCPP_INFO(node_->get_logger(), "Cluster %ld: ACCEPTED as valid circle", i);
                    
                    // 将恢复后的圆心坐标添加到点云中
                    pcl::PointXYZ center_point;
                    center_point.x = coefficients->values[0];
                    center_point.y = coefficients->values[1];
                    center_point.z = 0.0;
                    center_z0_cloud_->push_back(center_point);

                    // 将圆心坐标逆变换回原始坐标系
                    Eigen::Vector3d aligned_point(center_point.x, center_point.y, center_point.z + average_z);
                    Eigen::Vector3d original_point = R_inv * aligned_point;

                    pcl::PointXYZ center_point_origin;
                    center_point_origin.x = original_point.x();
                    center_point_origin.y = original_point.y();
                    center_point_origin.z = original_point.z();
                    center_cloud->points.push_back(center_point_origin);
                    
                    RCLCPP_INFO(node_->get_logger(), "Added circle center: (%.3f, %.3f, %.3f)", 
                               center_point_origin.x, center_point_origin.y, center_point_origin.z);
                    // 为接受的圆心添加特殊标记（白色大点）
                    for (int j = 0; j < 5; ++j) {  // 添加5个点形成较大的标记
                        pcl::PointXYZRGB center_marker;
                        center_marker.x = center_point_origin.x + (j-2) * 0.01;
                        center_marker.y = center_point_origin.y;
                        center_marker.z = center_point_origin.z;
                        center_marker.r = 255;
                        center_marker.g = 255;
                        center_marker.b = 255;
                        colored_clusters->push_back(center_marker);
                    }
                }
                else 
                {
                    RCLCPP_WARN(node_->get_logger(), "Cluster %ld: REJECTED due to high fitting error", i);
                }
            }
            else 
            {
                RCLCPP_WARN(node_->get_logger(), "Cluster %ld: No inliers found in circle fitting", i);
            }
        }
        
        RCLCPP_INFO(node_->get_logger(), "Final detected circle centers: %ld (expected 4)", center_cloud->size());
        
        // 保存中间结果点云用于调试
        std::string debug_dir = "debug_lidar/";
        system(("mkdir -p " + debug_dir).c_str());
        
        // 确保所有点云结构正确
        filtered_cloud_->width = filtered_cloud_->size();
        filtered_cloud_->height = 1;
        filtered_cloud_->is_dense = true;
        pcl::io::savePCDFileASCII(debug_dir + "01_filtered_cloud.pcd", *filtered_cloud_);
        
        plane_cloud_->width = plane_cloud_->size();
        plane_cloud_->height = 1;
        plane_cloud_->is_dense = true;
        pcl::io::savePCDFileASCII(debug_dir + "02_plane_cloud.pcd", *plane_cloud_);
        
        aligned_cloud_->width = aligned_cloud_->size();
        aligned_cloud_->height = 1;
        aligned_cloud_->is_dense = true;
        pcl::io::savePCDFileASCII(debug_dir + "03_aligned_cloud.pcd", *aligned_cloud_);
        
        edge_cloud_->width = edge_cloud_->size();
        edge_cloud_->height = 1;
        edge_cloud_->is_dense = true;
        pcl::io::savePCDFileASCII(debug_dir + "04_edge_cloud.pcd", *edge_cloud_);
        
        if (center_z0_cloud_->size() > 0) {
            center_z0_cloud_->width = center_z0_cloud_->size();
            center_z0_cloud_->height = 1;
            center_z0_cloud_->is_dense = true;
            pcl::io::savePCDFileASCII(debug_dir + "05_center_z0_cloud.pcd", *center_z0_cloud_);
        }
        
        // 保存检测到的圆心
        if (center_cloud->size() > 0) {
            center_cloud->width = center_cloud->size();
            center_cloud->height = 1;
            center_cloud->is_dense = true;
            pcl::io::savePCDFileASCII(debug_dir + "06_detected_centers.pcd", *center_cloud);
        }
        // 保存彩色点云
        if (colored_clusters->size() > 0) {
            colored_clusters->width = colored_clusters->size();
            colored_clusters->height = 1;
            colored_clusters->is_dense = true;
            pcl::io::savePCDFileASCII(debug_dir + "07_colored_clusters.pcd", *colored_clusters);
            RCLCPP_INFO(node_->get_logger(), "Saved colored clusters to: %s07_colored_clusters.pcd", debug_dir.c_str());
        }
        // 保存每个聚类的点云用于详细分析
        for (size_t i = 0; i < cluster_indices.size(); ++i) 
        {
            pcl::PointCloud<pcl::PointXYZ>::Ptr cluster(new pcl::PointCloud<pcl::PointXYZ>);
            for (const auto& idx : cluster_indices[i].indices) {
                cluster->push_back(edge_cloud_->points[idx]);
            }
            
            if (cluster->size() > 0) {
                cluster->width = cluster->size();
                cluster->height = 1;
                cluster->is_dense = true;
                std::string cluster_filename = debug_dir + "cluster_" + std::to_string(i) + ".pcd";
                pcl::io::savePCDFileASCII(cluster_filename, *cluster);
            }
        }
        
        RCLCPP_INFO(node_->get_logger(), "Saved debug point clouds to: %s", debug_dir.c_str());
    }

    // 获取中间结果的点云
    pcl::PointCloud<pcl::PointXYZ>::Ptr getFilteredCloud() const { return filtered_cloud_; }
    pcl::PointCloud<pcl::PointXYZ>::Ptr getPlaneCloud() const { return plane_cloud_; }
    pcl::PointCloud<pcl::PointXYZ>::Ptr getAlignedCloud() const { return aligned_cloud_; }
    pcl::PointCloud<pcl::PointXYZ>::Ptr getEdgeCloud() const { return edge_cloud_; }
    pcl::PointCloud<pcl::PointXYZ>::Ptr getCenterZ0Cloud() const { return center_z0_cloud_; }
};

typedef std::shared_ptr<LidarDetect> LidarDetectPtr;

#endif