#include <pcl/io/pcd_io.h>
#include <pcl/point_types.h>
#include <iostream>

int main(int argc, char **argv) {
  pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_1(new pcl::PointCloud<pcl::PointXYZ>);
  pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_2(new pcl::PointCloud<pcl::PointXYZ>);
  if (!pcl::io::loadPCDFile("/home/hjx/based_point_segment_ws/left_dense_pcd_2.pcd", *cloud_1)) {
    std::cout << "load cloud_1 success" << std::endl;
  }
  if (!pcl::io::loadPCDFile("/home/hjx/based_point_segment_ws/fix_right_pcd.pcd", *cloud_2)) {
    std::cout << "load cloud_2 success" << std::endl;
  }
  *cloud_1 += *cloud_2;
  pcl::io::savePCDFile("add.pcd", *cloud_1);
}