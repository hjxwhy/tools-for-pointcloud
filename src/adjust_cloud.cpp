//
// Created by hjx on 2020/9/28.
//
#include <iostream>
#include <vector>
#include <thread>
#include <mutex>
#include <pcl/point_types.h>
#include <pcl/io/pcd_io.h>
#include <pcl/search/search.h>
#include <pcl/search/kdtree.h>
#include <pcl/features/normal_3d.h>
#include <pcl/visualization/cloud_viewer.h>
#include <pcl/filters/passthrough.h>
#include <pcl/segmentation/region_growing.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl_conversions/pcl_conversions.h>
#include <pcl/segmentation/sac_segmentation.h>
#include <pcl/common/transforms.h>

#include <ros/ros.h>
#include <geometry_msgs/PointStamped.h>
#include <sensor_msgs/PointCloud2.h>
#include <visualization_msgs/Marker.h>

#include <Eigen/Core>
#include <Eigen/Dense>

#include <dynamic_reconfigure/server.h>
#include "region_growing_segmentation/cloud_tutorialsConfig.h"
#include <tf/transform_broadcaster.h>
#define PI 3.14
using namespace std;
class AdjustCloud {
 public:
  AdjustCloud(pcl::PointCloud<pcl::PointXYZ>::Ptr source) : source_cloud(source),roll(0),pitch(0),yaw(-43.2),x(52.0),y(28.0),z(-1.0) {
    pub = nh.advertise<sensor_msgs::PointCloud2>("/trans_pointcloud", 1);
    source_pub = nh.advertise<sensor_msgs::PointCloud2>("/source_pointcloud", 1);
    sub = nh.subscribe("/livox/lidar", 1, &AdjustCloud::pointCallback, this);
    f = boost::bind(&AdjustCloud::callback, this, _1, _2);
    server.setCallback(f);
  }
  void pointCallback(const sensor_msgs::PointCloud2ConstPtr &input) {
    //cout << "I RECEIVED INPUT !" << endl;
    std::lock_guard<std::mutex> lock(cloud_lock_);
    pcl::fromROSMsg(*input, *cloud_in);
  }

  void callback(region_growing_segmentation::cloud_tutorialsConfig &config, uint32_t level) {
    ROS_INFO("Reconfigure Request: %f %f %f",
             config.roll, config.pitch, config.yaw
    );
    std::lock_guard<std::mutex> lock(trans_lock_);
    roll = config.roll * PI / 180;
    pitch = config.pitch * PI / 180;
    yaw = config.yaw * PI / 180;
    x = config.x;
    y = config.y;
    z = config.z;
  }
  void transformCloud(Eigen::Matrix4d &transform);
  Eigen::Matrix4d getTransform();
  void pubTransformedCloud();
  bool loadSourceCloud();
  tf::Transform broadcasterTF();
 private:
  ros::NodeHandle nh;
  ros::Publisher pub;
  ros::Publisher source_pub;
  ros::Subscriber sub;
  tf::TransformBroadcaster br;
  dynamic_reconfigure::Server<region_growing_segmentation::cloud_tutorialsConfig> server;
  dynamic_reconfigure::Server<region_growing_segmentation::cloud_tutorialsConfig>::CallbackType f;
  pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_in{new pcl::PointCloud<pcl::PointXYZ>};//转换为pcl格式
  pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_final{new pcl::PointCloud<pcl::PointXYZ>};//加工后的pcl格式
  sensor_msgs::PointCloud2 cloud_out;//转换为ros格式
  sensor_msgs::PointCloud2 source_cloud_out;
  float roll, pitch, yaw = 0;
  float x, y, z = 0;
  std::mutex cloud_lock_;
  std::mutex trans_lock_;
  pcl::PointCloud<pcl::PointXYZ>::Ptr source_cloud;
};

Eigen::Matrix4d AdjustCloud::getTransform() {
  std::lock_guard<std::mutex> lock(trans_lock_);
  Eigen::Vector3d euler_angle(yaw, pitch,roll);
  Eigen::Vector3d trans_vector(x, y, z);
  Eigen::AngleAxisd rotation_vector;
  rotation_vector = Eigen::AngleAxisd(euler_angle[0], Eigen::Vector3d::UnitZ()) *
      Eigen::AngleAxisd(euler_angle[1], Eigen::Vector3d::UnitY()) *
      Eigen::AngleAxisd(euler_angle[2], Eigen::Vector3d::UnitX());
  Eigen::Isometry3d T = Eigen::Isometry3d::Identity();
  T.rotate(rotation_vector);
  T.pretranslate(trans_vector);
  Eigen::Matrix4d rotation_matrix = T.matrix();
  return rotation_matrix;
}
void AdjustCloud::transformCloud(Eigen::Matrix4d &transform) {
//  Eigen::Isometry3d T = Eigen::Isometry3d::Identity();
//  Eigen::Matrix4d matrix4d = T.matrix();
  if (!cloud_in->empty()) {
    std::lock_guard<std::mutex> lock(cloud_lock_);
    cloud_final->clear();
    pcl::transformPointCloud(*cloud_in, *cloud_final, transform);
  }
}
void AdjustCloud::pubTransformedCloud() {
  std::lock_guard<std::mutex> lock(cloud_lock_);
  ros::Time time = ros::Time::now();
  if (!cloud_final->empty()) {
    pcl::toROSMsg(*cloud_final, cloud_out);
    cloud_out.header.stamp = time;
    cloud_out.header.frame_id = "livox_frame";
    pub.publish(cloud_out);
    //cout << "I PROCESSED IT !" << endl;
  }
  if (!source_cloud->empty()) {
    pcl::toROSMsg(*source_cloud, source_cloud_out);
    source_cloud_out.header.stamp = time;
    source_cloud_out.header.frame_id = "livox_frame";
    source_pub.publish(source_cloud_out);
  }
}
bool AdjustCloud::loadSourceCloud() {
  if (pcl::io::loadPCDFile("/home/hjx/jiaotongting/sub_map.pcd", *source_cloud)) {
    std::cerr << "failed to load " << "source_pcd" << std::endl;
    // downsampling
    pcl::PointCloud<pcl::PointXYZ>::Ptr downsampled(new pcl::PointCloud<pcl::PointXYZ>());
    pcl::VoxelGrid<pcl::PointXYZ> voxelgrid;
    voxelgrid.setLeafSize(0.1f, 0.1f, 0.1f);
    voxelgrid.setInputCloud(source_cloud);
    voxelgrid.filter(*downsampled);
    *source_cloud = *downsampled;
    return 0;
  }
}
tf::Transform AdjustCloud::broadcasterTF() {
  tf::Transform transform;
  tf::Vector3 vector3(x, y, z);
  transform.setOrigin(vector3);
  tf::Quaternion q;
  q.setRPY(roll, pitch, yaw);
  transform.setRotation(q);
  this->br.sendTransform(tf::StampedTransform(transform,ros::Time::now(),"livox_frame","target_frame"));
  return transform;
}

int
main(int argc, char **argv) {
  ros::init(argc, argv, "AdjustCloud");
  pcl::PointCloud<pcl::PointXYZ>::Ptr source_ptr(new pcl::PointCloud<pcl::PointXYZ>());
  AdjustCloud trans_pointcloud(source_ptr);
  ros::Rate sleep_rate(5);
  trans_pointcloud.loadSourceCloud();

  Eigen::Vector3d euler_angle(-80 * PI / 180, 0,0);
  Eigen::Vector3d trans_vector(106, 12, -1);
  Eigen::AngleAxisd rotation_vector;
  rotation_vector = Eigen::AngleAxisd(euler_angle[0], Eigen::Vector3d::UnitZ()) *
      Eigen::AngleAxisd(euler_angle[1], Eigen::Vector3d::UnitY()) *
      Eigen::AngleAxisd(euler_angle[2], Eigen::Vector3d::UnitX());
  Eigen::Isometry3d T = Eigen::Isometry3d::Identity();
  T.rotate(rotation_vector);
  T.pretranslate(trans_vector);
  Eigen::Matrix4d rotation_matrix = T.matrix();

  while (ros::ok()) {
    trans_pointcloud.broadcasterTF();

    Eigen::Matrix4d transform = trans_pointcloud.getTransform();
    trans_pointcloud.transformCloud(transform);
    //trans_pointcloud.transformCloud(rotation_matrix);

    trans_pointcloud.pubTransformedCloud();
    sleep_rate.sleep();
    ros::spinOnce();
  }
}
