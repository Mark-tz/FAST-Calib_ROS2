/* 
Developer: Chunran Zheng <zhengcr@connect.hku.hk>

This file is subject to the terms and conditions outlined in the 'LICENSE' file,
which is included as part of this source code package.
*/

#ifndef DATA_PREPROCESS_HPP
#define DATA_PREPROCESS_HPP

#include <sensor_msgs/msg/point_cloud2.hpp>
#include <livox_ros_driver/msg/custom_msg.hpp>
#include <Eigen/Core>
#include <pcl/io/pcd_io.h>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl_conversions/pcl_conversions.h>
#include <rclcpp/rclcpp.hpp>
#include <rosbag2_cpp/readers/sequential_reader.hpp>
#include <rosbag2_storage/storage_options.hpp>
#include <rosbag2_cpp/converter_interfaces/serialization_format_converter.hpp>
#include <rclcpp/serialization.hpp>

using namespace std;
using namespace cv;
using namespace pcl;

class DataPreprocess
{
public:
    cv::Mat img_input_;
    pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_input_;
    
    DataPreprocess(Params &params)
        : cloud_input_(new pcl::PointCloud<pcl::PointXYZ>)
    {
        // 读取图像
        img_input_ = cv::imread(params.image_path);
        if (img_input_.empty()) {
            RCLCPP_ERROR(rclcpp::get_logger("data_preprocess"), "Failed to load image: %s", params.image_path.c_str());
            return;
        }
        RCLCPP_INFO(rclcpp::get_logger("data_preprocess"), "Loaded image: %s", params.image_path.c_str());
        
        // 读取rosbag2数据
        loadPointCloudFromBag(params.bag_path, params.lidar_topic);
    }
    
private:
    void loadPointCloudFromBag(const std::string& bag_path, const std::string& lidar_topic)
    {
        rosbag2_cpp::readers::SequentialReader reader;
        rosbag2_storage::StorageOptions storage_options;
        storage_options.uri = bag_path;
        storage_options.storage_id = "sqlite3";
        
        rosbag2_cpp::ConverterOptions converter_options;
        converter_options.input_serialization_format = "cdr";
        converter_options.output_serialization_format = "cdr";
        
        try {
            reader.open(storage_options, converter_options);
            
            // 获取topic信息来判断消息类型
            auto topics_and_types = reader.get_all_topics_and_types();
            std::string message_type;
            for (const auto& topic_info : topics_and_types) {
                if (topic_info.name == lidar_topic) {
                    message_type = topic_info.type;
                    break;
                }
            }
            
            while (reader.has_next()) {
                auto bag_message = reader.read_next();
                
                if (bag_message->topic_name != lidar_topic) {
                    continue;
                }
                
                // 根据实际消息类型处理
                if (message_type == "livox_ros_driver/msg/CustomMsg") {
                    // Handle Livox custom message for ROS2
                    auto livox_msg = std::make_shared<livox_ros_driver::msg::CustomMsg>();
                    rclcpp::Serialization<livox_ros_driver::msg::CustomMsg> serialization;
                    rclcpp::SerializedMessage serialized_msg(*bag_message->serialized_data);
                    serialization.deserialize_message(&serialized_msg, livox_msg.get());
                    
                    cloud_input_->reserve(livox_msg->point_num);
                    for (uint i = 0; i < livox_msg->point_num; ++i) {
                        pcl::PointXYZ p;
                        p.x = livox_msg->points[i].x;
                        p.y = livox_msg->points[i].y;
                        p.z = livox_msg->points[i].z;
                        cloud_input_->points.push_back(p);
                    }
                } else if (message_type == "sensor_msgs/msg/PointCloud2") {
                    // Handle standard PointCloud2 message
                    auto pcl_msg = std::make_shared<sensor_msgs::msg::PointCloud2>();
                    rclcpp::Serialization<sensor_msgs::msg::PointCloud2> serialization;
                    rclcpp::SerializedMessage serialized_msg(*bag_message->serialized_data);
                    serialization.deserialize_message(&serialized_msg, pcl_msg.get());
                    
                    pcl::PointCloud<pcl::PointXYZ> temp_cloud;
                    pcl::fromROSMsg(*pcl_msg, temp_cloud);
                    *cloud_input_ += temp_cloud;
                }
            }
            
            RCLCPP_INFO(rclcpp::get_logger("data_preprocess"), "Loaded %ld points from the rosbag.", cloud_input_->size());
        } catch (const std::exception& e) {
            RCLCPP_ERROR(rclcpp::get_logger("data_preprocess"), "Error reading bag file: %s", e.what());
        }
    }
};

typedef std::shared_ptr<DataPreprocess> DataPreprocessPtr;

#endif