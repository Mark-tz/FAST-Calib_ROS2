/* 
Developer: Chunran Zheng <zhengcr@connect.hku.hk>

This file is subject to the terms and conditions outlined in the 'LICENSE' file,
which is included as part of this source code package.
*/

#include "qr_detect.hpp"
#include "lidar_detect.hpp"
#include "data_preprocess.hpp"
#include <rclcpp/rclcpp.hpp>

// 在main函数中添加更详细的调试输出
int main(int argc, char **argv)
{
    rclcpp::init(argc, argv);
    auto node = rclcpp::Node::make_shared("mono_qr_pattern");

    // 读取参数
    Params params = loadParameters(node);

    // 初始化 QR 检测和 LiDAR 检测
    QRDetectPtr qrDetectPtr;
    qrDetectPtr.reset(new QRDetect(node, params));

    LidarDetectPtr lidarDetectPtr;
    lidarDetectPtr.reset(new LidarDetect(node, params));

    DataPreprocessPtr dataPreprocessPtr;
    dataPreprocessPtr.reset(new DataPreprocess(params));

    // 读取图像和点云
    cv::Mat img_input = dataPreprocessPtr->img_input_;
    pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_input = dataPreprocessPtr->cloud_input_;
    
    // 检测 QR 码
    PointCloud<PointXYZ>::Ptr qr_center_cloud(new PointCloud<PointXYZ>);
    qr_center_cloud->reserve(4);
    qrDetectPtr->detect_qr(img_input, qr_center_cloud);
    std::cout << BOLDGREEN << "[Debug] QR detection found " << qr_center_cloud->size() << " centers" << RESET << std::endl;
    for (size_t i = 0; i < qr_center_cloud->size(); ++i) {
        std::cout << BOLDCYAN << "  QR center " << i << ": (" 
                  << qr_center_cloud->points[i].x << ", " 
                  << qr_center_cloud->points[i].y << ", " 
                  << qr_center_cloud->points[i].z << ")" << RESET << std::endl;
    }

    // 检测 LiDAR 数据
    PointCloud<PointXYZ>::Ptr lidar_center_cloud(new PointCloud<PointXYZ>);
    lidar_center_cloud->reserve(4);
    lidarDetectPtr->detect_lidar(cloud_input, lidar_center_cloud);
    std::cout << BOLDGREEN << "[Debug] LiDAR detection found " << lidar_center_cloud->size() << " centers" << RESET << std::endl;
    for (size_t i = 0; i < lidar_center_cloud->size(); ++i) {
        std::cout << BOLDCYAN << "  LiDAR center " << i << ": (" 
                  << lidar_center_cloud->points[i].x << ", " 
                  << lidar_center_cloud->points[i].y << ", " 
                  << lidar_center_cloud->points[i].z << ")" << RESET << std::endl;
    }

    // 保存检测结果用于调试
    if (qr_center_cloud->size() > 0) {
        // 确保点云结构正确
        qr_center_cloud->width = qr_center_cloud->size();
        qr_center_cloud->height = 1;
        qr_center_cloud->is_dense = true;
        pcl::io::savePCDFileASCII("debug_qr_centers.pcd", *qr_center_cloud);
        std::cout << BOLDGREEN << "[Debug] Saved QR centers to debug_qr_centers.pcd" << RESET << std::endl;
    }
    if (lidar_center_cloud->size() > 0) {
        // 确保点云结构正确
        lidar_center_cloud->width = lidar_center_cloud->size();
        lidar_center_cloud->height = 1;
        lidar_center_cloud->is_dense = true;
        pcl::io::savePCDFileASCII("debug_lidar_centers.pcd", *lidar_center_cloud);
        std::cout << BOLDGREEN << "[Debug] Saved LiDAR centers to debug_lidar_centers.pcd" << RESET << std::endl;
    }

    // 对 QR 和 LiDAR 检测到的圆心进行排序
    PointCloud<PointXYZ>::Ptr qr_centers(new PointCloud<PointXYZ>);
    PointCloud<PointXYZ>::Ptr lidar_centers(new PointCloud<PointXYZ>);
    
    std::cout << BOLDYELLOW << "[Debug] Sorting QR centers..." << RESET << std::endl;
    sortPatternCenters(qr_center_cloud, qr_centers, "camera");
    std::cout << BOLDYELLOW << "[Debug] Sorting LiDAR centers..." << RESET << std::endl;
    sortPatternCenters(lidar_center_cloud, lidar_centers, "lidar");

    std::cout << BOLDGREEN << "[Debug] After sorting - QR: " << qr_centers->size() 
              << " centers, LiDAR: " << lidar_centers->size() << " centers" << RESET << std::endl;

    // 保存排序后的结果
    if (qr_centers->size() > 0) {
        qr_centers->width = qr_centers->size();
        qr_centers->height = 1;
        qr_centers->is_dense = true;
        pcl::io::savePCDFileASCII("debug_qr_centers_sorted.pcd", *qr_centers);
        std::cout << BOLDGREEN << "[Debug] Saved sorted QR centers to debug_qr_centers_sorted.pcd" << RESET << std::endl;
    }
    if (lidar_centers->size() > 0) {
        lidar_centers->width = lidar_centers->size();
        lidar_centers->height = 1;
        lidar_centers->is_dense = true;
        pcl::io::savePCDFileASCII("debug_lidar_centers_sorted.pcd", *lidar_centers);
        std::cout << BOLDGREEN << "[Debug] Saved sorted LiDAR centers to debug_lidar_centers_sorted.pcd" << RESET << std::endl;
    }

    // 保存中间结果：排序后的 LiDAR 圆心和 QR 圆心
    saveTargetHoleCenters(lidar_centers, qr_centers, params);

    // 计算外参
    Eigen::Matrix4f transformation;
    pcl::registration::TransformationEstimationSVD<pcl::PointXYZ, pcl::PointXYZ> svd;
    svd.estimateRigidTransformation(*lidar_centers, *qr_centers, transformation);

    // 将 LiDAR 点云转换到 QR 码坐标系
    pcl::PointCloud<pcl::PointXYZ>::Ptr aligned_lidar_centers(new pcl::PointCloud<pcl::PointXYZ>);
    aligned_lidar_centers->reserve(lidar_centers->size());
    alignPointCloud(lidar_centers, aligned_lidar_centers, transformation);
    
    double rmse = computeRMSE(qr_centers, aligned_lidar_centers);
    if (rmse > 0) 
    {
      std::cout << BOLDYELLOW << "[Result] RMSE: " << BOLDRED << std::fixed << std::setprecision(4)
      << rmse << " m" << RESET << std::endl;
    }

    std::cout << BOLDYELLOW << "[Result] Single-scene calibration: extrinsic parameters T_cam_lidar = " << RESET << std::endl;
    std::cout << BOLDCYAN << std::fixed << std::setprecision(6) << transformation << RESET << std::endl;

    pcl::PointCloud<pcl::PointXYZRGB>::Ptr colored_cloud(new pcl::PointCloud<pcl::PointXYZRGB>);
    projectPointCloudToImage(cloud_input, transformation, qrDetectPtr->cameraMatrix_, qrDetectPtr->distCoeffs_, img_input, colored_cloud);

    saveCalibrationResults(params, transformation, colored_cloud, qrDetectPtr->imageCopy_);

    auto colored_cloud_pub = node->create_publisher<sensor_msgs::msg::PointCloud2>("colored_cloud", 1);
    auto aligned_lidar_centers_pub = node->create_publisher<sensor_msgs::msg::PointCloud2>("aligned_lidar_centers", 1);

    // 主循环
    rclcpp::Rate rate(1);
    while (rclcpp::ok()) 
    {
      if (DEBUG) 
      {
        // 发布 QR 检测结果
        sensor_msgs::msg::PointCloud2 qr_centers_msg;
        pcl::toROSMsg(*qr_centers, qr_centers_msg);
        qr_centers_msg.header.stamp = node->get_clock()->now();
        qr_centers_msg.header.frame_id = "map";
        qrDetectPtr->qr_pub_->publish(qr_centers_msg);

        // 发布 LiDAR 检测结果
        sensor_msgs::msg::PointCloud2 lidar_centers_msg;
        pcl::toROSMsg(*lidar_centers, lidar_centers_msg);
        lidar_centers_msg.header = qr_centers_msg.header;
        lidarDetectPtr->center_pub_->publish(lidar_centers_msg);

        // 发布中间结果
        sensor_msgs::msg::PointCloud2 filtered_cloud_msg;
        pcl::toROSMsg(*lidarDetectPtr->getFilteredCloud(), filtered_cloud_msg);
        filtered_cloud_msg.header = qr_centers_msg.header;
        lidarDetectPtr->filtered_pub_->publish(filtered_cloud_msg);

        sensor_msgs::msg::PointCloud2 plane_cloud_msg;
        pcl::toROSMsg(*lidarDetectPtr->getPlaneCloud(), plane_cloud_msg);
        plane_cloud_msg.header = qr_centers_msg.header;
        lidarDetectPtr->plane_pub_->publish(plane_cloud_msg);

        sensor_msgs::msg::PointCloud2 aligned_cloud_msg;
        pcl::toROSMsg(*lidarDetectPtr->getAlignedCloud(), aligned_cloud_msg);
        aligned_cloud_msg.header = qr_centers_msg.header;
        lidarDetectPtr->aligned_pub_->publish(aligned_cloud_msg);

        sensor_msgs::msg::PointCloud2 edge_cloud_msg;
        pcl::toROSMsg(*lidarDetectPtr->getEdgeCloud(), edge_cloud_msg);
        edge_cloud_msg.header = qr_centers_msg.header;
        lidarDetectPtr->edge_pub_->publish(edge_cloud_msg);

        sensor_msgs::msg::PointCloud2 lidar_centers_z0_msg;
        pcl::toROSMsg(*lidarDetectPtr->getCenterZ0Cloud(), lidar_centers_z0_msg);
        lidar_centers_z0_msg.header = qr_centers_msg.header;
        lidarDetectPtr->center_z0_pub_->publish(lidar_centers_z0_msg);

        // 发布外参变换后的LiDAR点云
        sensor_msgs::msg::PointCloud2 aligned_lidar_centers_msg;
        pcl::toROSMsg(*aligned_lidar_centers, aligned_lidar_centers_msg);
        aligned_lidar_centers_msg.header = qr_centers_msg.header;
        aligned_lidar_centers_pub->publish(aligned_lidar_centers_msg);

        // 发布彩色点云
        sensor_msgs::msg::PointCloud2 colored_cloud_msg;
        pcl::toROSMsg(*colored_cloud, colored_cloud_msg);
        colored_cloud_msg.header = qr_centers_msg.header;
        colored_cloud_pub->publish(colored_cloud_msg);
      }
      rclcpp::spin_some(node);
      rate.sleep();
    }

    rclcpp::shutdown();

    return 0;
}