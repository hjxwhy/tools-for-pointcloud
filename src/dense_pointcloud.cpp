//
// Created by hjx on 2020/10/14.
//
#include <iostream>
#include <vector>
#include <thread>
#include <boost/circular_buffer.hpp>
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
#include <ros/ros.h>
#include <geometry_msgs/PointStamped.h>
#include <sensor_msgs/PointCloud2.h>
//std::vector<pcl::PointCloud<pcl::PointXYZ>> dense_cloud;
boost::circular_buffer<pcl::PointCloud<pcl::PointXYZ> > circular_buffer_cloud(20);
pcl::PointCloud<pcl::PointXYZ> dense_cloud;
bool save_flag = false;
void callback(const sensor_msgs::PointCloud2Ptr &point_msg) {
  if (!point_msg->data.empty()) {
    pcl::PointCloud<pcl::PointXYZ> cloud;
    pcl::fromROSMsg(*point_msg, cloud);
    circular_buffer_cloud.push_back(cloud);
  }
}

void pubDenseCloud(ros::Publisher &pub) {
  if (!circular_buffer_cloud.empty()) {
    dense_cloud.clear();
    for (auto it = circular_buffer_cloud.begin(); it != circular_buffer_cloud.end(); it++) {
      dense_cloud += *it;
    }
    if (circular_buffer_cloud.size() == 20 && !save_flag){
      pcl::io::savePCDFile("dense_pcd.pcd",dense_cloud);
      save_flag = true;
    }
    sensor_msgs::PointCloud2 out_cloud;
    pcl::toROSMsg(dense_cloud, out_cloud);
    out_cloud.header.frame_id = "livox_frame";
    out_cloud.header.stamp = ros::Time::now();
    pub.publish(out_cloud);
  }
}

void savePointCloud() {

}
int main(int argc, char **argv) {
  ros::init(argc, argv, "dense_pointcloud_node");
  ros::NodeHandle nh;
  ros::Subscriber raw_point_sub = nh.subscribe("/aligned_pointcloud", 10, callback);
  ros::Publisher dense_cloud_pub = nh.advertise<sensor_msgs::PointCloud2>("dense_cloud", 1);
//  pcl::PointCloud<pcl::PointXYZ>::Ptr cloud(new pcl::PointCloud<pcl::PointXYZ>);
//  pcl::io::loadPCDFile("/home/hjx/based_point_segment_ws/right_dense.pcd",*cloud);
//  pcl::PointCloud<pcl::PointXYZ>::Ptr downsampled(new pcl::PointCloud<pcl::PointXYZ>());
//  pcl::VoxelGrid<pcl::PointXYZ> voxelgrid;
//  voxelgrid.setLeafSize(1.5f, 1.5f, 1.5f);
//  voxelgrid.setInputCloud(cloud);
//  voxelgrid.filter(*downsampled);
//  *cloud = *downsampled;
//  cout << "cloud size " << cloud->size() << endl;
//  for (auto it = cloud->points.begin(); it != cloud->points.end();)
//  {
//    std::cout << *it << std::endl;
//    cloud->erase(it);
//    std::cout << *it << std::endl;
//    std::cout << std::endl;
//  }
//  cout << "cloud size " << cloud->size() << endl;

  ros::Rate sleep_rate(10);
  while (ros::ok()) {
    pubDenseCloud(dense_cloud_pub);
    sleep_rate.sleep();
    ros::spinOnce();
  }
}