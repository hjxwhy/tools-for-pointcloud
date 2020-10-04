#include <iostream>
#include <vector>
#include <thread>
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
//#include <pcl/ModelCoefficients.h>
//#include <pcl/sample_consensus/method_types.h>
//#include <pcl/sample_consensus/model_types.h>

#include <ros/ros.h>
#include <geometry_msgs/PointStamped.h>
#include <sensor_msgs/PointCloud2.h>
#include <visualization_msgs/Marker.h>

#include <Eigen/Core>
#include <Eigen/Dense>
#include <dynamic_reconfigure/server.h>
#include "region_growing_segmentation/TutorialsConfig.h"

pcl::PointXYZ click_point;
bool new_point = false;
Eigen::Vector3f marker_normal(0, 1, 0);

float bias_x,bias_y,bias_z,bias_q_x,bias_q_y,bias_q_z,bias_q_w = 0;
float bias_s_x,bias_s_z = 0;
//pcl::visualization::CloudViewer viewer("Cluster viewer");

void callback(region_growing_segmentation::TutorialsConfig &config, uint32_t level) {
  ROS_INFO("Reconfigure Request: %f %f",
           config.position_x,config.position_y
  );

  bias_x = config.position_x;
  bias_y = config.position_y;
  bias_z = config.position_z;
  bias_q_x = config.quaternion_x;
  bias_q_y = config.quaternion_y;
  bias_q_z = config.quaternion_z;
  bias_q_w = config.quaternion_w;
  bias_s_x = config.scale_x;
  bias_s_z = config.scale_z;
}

void pointCallback(const geometry_msgs::PointStampedPtr &msg) {
  std::cout << "new point" << std::endl;
  click_point.x = msg->point.x;
  click_point.y = msg->point.y;
  click_point.z = msg->point.z;
  new_point = true;
}

//void runViewer(){
//  while (!viewer.wasStopped()) {
//  }
//}
Eigen::Vector3f crossProduct(Eigen::Vector3f a, Eigen::Vector3f b) {
  Eigen::Vector3f c;

  c[0] = a[1] * b[2] - a[2] * b[1];
  c[1] = a[2] * b[0] - a[0] * b[2];
  c[2] = a[0] * b[1] - a[1] * b[0];

  return c;
}

float dotProduct(Eigen::Vector3f a, Eigen::Vector3f b) {
  float result;
  result = a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
  return result;
}

float normalize(Eigen::Vector3f v) {
  float result;
  result = sqrt(v[0] * v[0] + v[1] * v[1] + v[2] * v[2]);
  return result;
}

Eigen::Matrix3f rotationMatrix(float angle, Eigen::Vector3f u) {
  float norm = normalize(u);
  Eigen::Matrix3f rotatinMatrix;

  u(0) = u(0) / norm;
  u(1) = u(1) / norm;
  u(2) = u(2) / norm;

  rotatinMatrix(0, 0) = cos(angle) + u(0) * u(0) * (1 - cos(angle));
  rotatinMatrix(0, 1) = u(0) * u(1) * (1 - cos(angle)) - u(2) * sin(angle);
  rotatinMatrix(0, 2) = u(1) * sin(angle) + u(0) * u(2) * (1 - cos(angle));

  rotatinMatrix(1, 0) = u(2) * sin(angle) + u(0) * u(1) * (1 - cos(angle));
  rotatinMatrix(1, 1) = cos(angle) + u(1) * u(1) * (1 - cos(angle));
  rotatinMatrix(1, 2) = -u(0) * sin(angle) + u(1) * u(2) * (1 - cos(angle));

  rotatinMatrix(2, 0) = -u(1) * sin(angle) + u(0) * u(2) * (1 - cos(angle));
  rotatinMatrix(2, 1) = u(0) * sin(angle) + u(1) * u(2) * (1 - cos(angle));
  rotatinMatrix(2, 2) = cos(angle) + u(2) * u(2) * (1 - cos(angle));

  return rotatinMatrix;
}
Eigen::Matrix3f calculationNormalRotation(Eigen::Vector3f plane_normal) {
  Eigen::Vector3f rotation_axis;
  float rotation_angle;
  Eigen::Matrix3f rotation_matrix;
  rotation_axis = crossProduct(marker_normal, plane_normal);
  rotation_angle = acos(dotProduct(marker_normal, plane_normal) / normalize(marker_normal) / normalize(plane_normal));
  rotation_matrix = rotationMatrix(rotation_angle, rotation_axis);
  return rotation_matrix;
}

visualization_msgs::Marker drawMarker(Eigen::Vector3f plane_normal, pcl::PointXYZ min_point) {
  Eigen::Matrix3f rotation_matrix = calculationNormalRotation(plane_normal);
  Eigen::Quaternionf q(rotation_matrix);
  visualization_msgs::Marker marker;
  marker.header.frame_id = "cloud";
  marker.header.stamp = ros::Time();
  marker.ns = "/";
  marker.id = 0;
  marker.type = visualization_msgs::Marker::CUBE;
  marker.action = visualization_msgs::Marker::ADD;
  marker.pose.position.x = min_point.x + bias_x;
  marker.pose.position.y = min_point.y + bias_y;
  marker.pose.position.z = min_point.z + bias_z;
  marker.pose.orientation.x = q.x() + bias_q_x;
  marker.pose.orientation.y = q.y() + bias_q_y;
  marker.pose.orientation.z = q.z() + bias_q_z;
  marker.pose.orientation.w = q.w() + bias_q_w;
  //std::cout << "quaternion x y z :" << q.x() << " " << q.y() << " " << q.z() << std::endl;
  marker.scale.x = 10  + bias_s_x;
  marker.scale.y = 0.1;
  marker.scale.z = 10 + bias_s_z;
  marker.color.a = 1.0; // Don't forget to set the alpha!
  marker.color.r = 0.0;
  marker.color.g = 1.0;
  marker.color.b = 0.0;
  return marker;
}

pcl::PointXYZ minDisPoint(pcl::PointCloud<pcl::PointXYZ> result_point) {
  int min_dis = 5000;
  int temp_dis = 0;
  int point_index = 0;
  for (int i = 0; i < result_point.points.size(); i++) {
    temp_dis = abs(0.099276 * result_point.points[i].x - 0.994441 * result_point.points[i].y
                       + 0.03509592 * result_point.points[i].z + 14.3107);
    if (temp_dis < min_dis) {
      min_dis = temp_dis;
      point_index = i;
    }
  }
  std::cout << "min distance: " << min_dis << std::endl;
  return result_point.points[point_index];
}

int
main(int argc, char **argv) {
  ros::init(argc, argv, "region_growing_segmention");

  dynamic_reconfigure::Server<region_growing_segmentation::TutorialsConfig> server;
  dynamic_reconfigure::Server<region_growing_segmentation::TutorialsConfig>::CallbackType f;

  f = boost::bind(&callback, _1, _2);
  server.setCallback(f);

  ros::NodeHandle nh;
  ros::Subscriber point_sub = nh.subscribe("/clicked_point", 1, pointCallback);
  ros::Publisher seg_cloud_pub = nh.advertise<sensor_msgs::PointCloud2>("/seg_pointcloud", 1);
  ros::Publisher cloud_pub = nh.advertise<sensor_msgs::PointCloud2>("/pointcloud", 1);
  ros::Publisher vis_pub = nh.advertise<visualization_msgs::Marker>("visualization_marker", 0);
  pcl::PointCloud<pcl::PointXYZ>::Ptr cloud(new pcl::PointCloud<pcl::PointXYZ>);
  if (pcl::io::loadPCDFile<pcl::PointXYZ>("/home/hjx/Documents/sub_map.pcd", *cloud) == -1) {
    std::cout << "Cloud reading failed." << std::endl;
    return (-1);
  }

  pcl::PointCloud<pcl::PointXYZ>::Ptr downsampled(new pcl::PointCloud<pcl::PointXYZ>());
  pcl::VoxelGrid<pcl::PointXYZ> voxelgrid;
  voxelgrid.setLeafSize(0.1f, 0.1f, 0.1f);
  voxelgrid.setInputCloud(cloud);
  voxelgrid.filter(*downsampled);
  *cloud = *downsampled;

  sensor_msgs::PointCloud2 out_pointcloud;
  pcl::toROSMsg(*cloud, out_pointcloud);
  out_pointcloud.header.frame_id = "cloud";
  //cloud_pub.publish(out_pointcloud);

  pcl::search::Search<pcl::PointXYZ>::Ptr tree(new pcl::search::KdTree<pcl::PointXYZ>);
  pcl::PointCloud<pcl::Normal>::Ptr normals(new pcl::PointCloud<pcl::Normal>);
  pcl::NormalEstimation<pcl::PointXYZ, pcl::Normal> normal_estimator;
  normal_estimator.setSearchMethod(tree);
  normal_estimator.setInputCloud(cloud);
  normal_estimator.setKSearch(50);
  normal_estimator.compute(*normals);

//  pcl::IndicesPtr indices(new std::vector<int>);
//  pcl::PassThrough<pcl::PointXYZ> pass;
//  pass.setInputCloud(cloud);
//  pass.setFilterFieldName("z");
//  pass.setFilterLimits(0.0, 1.0);
//  pass.filter(*indices);

  pcl::RegionGrowing<pcl::PointXYZ, pcl::Normal> reg;
  reg.setMinClusterSize(50);
  reg.setMaxClusterSize(1000000);
  reg.setSearchMethod(tree);
  reg.setNumberOfNeighbours(30);
  reg.setInputCloud(cloud);
  //reg.setIndices (indices);
  reg.setInputNormals(normals);
  reg.setSmoothnessThreshold(3.0 / 180.0 * M_PI);
  reg.setCurvatureThreshold(1.0);
  pcl::PointIndices point_cluster;

  std::vector<int> index;
  std::vector<float> sqr_distance;
  pcl::PointCloud<pcl::PointXYZRGB>::Ptr colored_cloud(new pcl::PointCloud<pcl::PointXYZRGB>);
  pcl::PointCloud<pcl::PointXYZ> result_point;
  sensor_msgs::PointCloud2 seg_out_pointcloud;
  visualization_msgs::Marker marker;
  pcl::PointXYZ min_point = click_point;
  Eigen::Vector3f plane_normal(0,1,0);
  ros::Rate sleep_rate(10);
  while (ros::ok()) {
    cloud_pub.publish(out_pointcloud);
    if (new_point) {
      tree->nearestKSearch(click_point, 1, index, sqr_distance);
      reg.getSegmentFromPoint(index[0], point_cluster);
      result_point.clear();
      for (std::vector<int>::iterator it = point_cluster.indices.begin(); it != point_cluster.indices.end(); it++) {
        result_point.points.push_back(cloud->points[*it]);
      }

      pcl::ModelCoefficients::Ptr coefficients(new pcl::ModelCoefficients);
      pcl::PointIndices::Ptr inliers(new pcl::PointIndices);

      pcl::SACSegmentation<pcl::PointXYZ> seg;

      seg.setOptimizeCoefficients(true);

      seg.setModelType(pcl::SACMODEL_PLANE);
      seg.setMethodType(pcl::SAC_RANSAC);
      seg.setDistanceThreshold(0.01);
      seg.setInputCloud(result_point.makeShared());
      seg.segment(*inliers, *coefficients);
      if (inliers->indices.size() == 0) {
        PCL_ERROR ("Could not estimate a planar model for the given dataset.");
        return (-1);
      }
      std::cerr << "Model coefficients: " << coefficients->values[0] << " "
                << coefficients->values[1] << " "
                << coefficients->values[2] << " "
                << coefficients->values[3] << std::endl;
      plane_normal << coefficients->values[0], coefficients->values[1], coefficients->values[2];
      min_point = minDisPoint(result_point);


//      std::vector<pcl::PointIndices> clusters;
//      reg.extract(clusters);
//
//      std::cout << "Number of clusters is equal to " << clusters.size() << std::endl;
//      std::cout << "First cluster has " << clusters[0].indices.size() << " points." << std::endl;
//      std::cout << "These are the indices of the points of the initial" <<
//                std::endl << "cloud that belong to the first cluster:" << std::endl;
//      int counter = 0;
//      while (counter < clusters[0].indices.size()) {
//        std::cout << clusters[0].indices[counter] << ", ";
//        counter++;
//        if (counter % 10 == 0)
//          std::cout << std::endl;
//      }
//      std::cout << std::endl;

//      pcl::PointCloud<pcl::PointXYZRGB>::Ptr colored_cloud = reg.getColoredCloud();
      //colored_cloud = reg.getColoredCloud();
      //pcl::io::savePCDFileASCII("colored_cloud.pcd", result_point);
      new_point = false;
    }
    marker = drawMarker(plane_normal, min_point);

    vis_pub.publish(marker);

//    pcl::visualization::CloudViewer viewer("Cluster viewer");
    if (!result_point.empty()) {
      pcl::toROSMsg(result_point, seg_out_pointcloud);
      seg_out_pointcloud.header.frame_id = "cloud";
      seg_cloud_pub.publish(seg_out_pointcloud);
    }
    ros::spinOnce();
    sleep_rate.sleep();

  }
//  while (!viewer.wasStopped()) {
//  }

  return (0);
}
