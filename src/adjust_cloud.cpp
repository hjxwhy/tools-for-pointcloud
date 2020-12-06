//
// Created by hjx on 2020/9/28.
//
#include <iostream>
#include <vector>
#include <thread>
#include <fstream>
#include <mutex>
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
#include <pcl/filters/statistical_outlier_removal.h>
#include <pcl_conversions/pcl_conversions.h>
#include <pcl/segmentation/sac_segmentation.h>
#include <pcl/common/transforms.h>
#include <pcl/common/common.h>
#include <pcl/surface/concave_hull.h>
#include <pcl/filters/project_inliers.h>
#include <pcl/kdtree/kdtree.h>
#include <pcl/segmentation/extract_clusters.h>
#include <visualization_msgs/Marker.h>

#include <ros/ros.h>
#include <geometry_msgs/PointStamped.h>
#include <sensor_msgs/PointCloud2.h>
#include <visualization_msgs/Marker.h>
#include <tf/transform_broadcaster.h>

#include <Eigen/Core>
#include <Eigen/Dense>
#include <dynamic_reconfigure/server.h>
#include <opencv2/opencv.hpp>
#include <boost/circular_buffer.hpp>
#include "region_growing_segmentation/cloud_tutorialsConfig.h"
#include "pclomp/ndt_omp.h"
#define PI 3.14
using namespace std;
using namespace cv;
struct Point2Pixel {
  int x, y;
  //int num;
  float hight;
};
class AdjustCloud {
 public:
  AdjustCloud()
      : roll(0), pitch(0), yaw(0), x(0), y(0), z(0), left_cir_buf_cloud(20), right_cir_buf_cloud(20) {
    pub = nh.advertise<sensor_msgs::PointCloud2>("/trans_pointcloud", 1);
    source_pub = nh.advertise<sensor_msgs::PointCloud2>("/source_pointcloud", 1);
    align_pub = nh.advertise<sensor_msgs::PointCloud2>("/aligned_pointcloud", 1);
    clustered_pub = nh.advertise<sensor_msgs::PointCloud2>("/clustered_pointcloud", 1);
    filtered_pub = nh.advertise<sensor_msgs::PointCloud2>("/filtered_pointcloud", 1);
    cloud_in_pub = nh.advertise<sensor_msgs::PointCloud2>("/cloud_in_pointcloud", 1);
    marker_pub = nh.advertise<visualization_msgs::Marker>("/visualization_marker", 1);

    left_sub = nh.subscribe("/livox/left_lidar", 1, &AdjustCloud::leftPointCallback, this);
    right_sub = nh.subscribe("/livox/right_lidar", 1, &AdjustCloud::rightPointCallback, this);
    f = boost::bind(&AdjustCloud::callback, this, _1, _2);
    server.setCallback(f);

    nh_private = ros::NodeHandle("~");
    nh_private.param<double>("NDT_TransformationEpsilon", ndt_transformation_epsilon_, 0.001);
    nh_private.param<double>("NDT_StepSize", ndt_step_size_, 0.1);
    nh_private.param<double>("NDT_Resolution", ndt_resolution_, 1);
    nh_private.param<int>("NDT_MaximumIterations", ndt_maximum_iterations_, 30);
    nh_private.param<double>("NDT_OulierRatio", ndt_oulier_ratio_, 0.2);

    nh_private.getParam("left_plane", left_plane_params);
    nh_private.getParam("right_plane", right_plane_params);
    nh_private.getParam("back_plane", back_plane_params);
    nh_private.getParam("YPR", ypr_params_);
    nh_private.getParam("xyz", xyz_params_);
    nh_private.getParam("lidar_YPR", lidar_ypr_params_);
    nh_private.getParam("lidar_xyz", lidar_xyz_params_);
    nh_private.getParam("left_2_right_YPR", left_2_right_YPR_);
    nh_private.getParam("left_2_right_xyz", left_2_right_xyz_);
    nh_private.getParam("load_map", load_map);
    nh_private.getParam("add_cloud", add_cloud);
    nh_private.getParam("hand_adjust", hand_adjust);
    nh_private.getParam("align_", align_);
    ndt.setMaximumIterations(ndt_maximum_iterations_);
    ndt.setNumThreads(omp_get_num_threads());
    ndt.setNeighborhoodSearchMethod(pclomp::DIRECT7);
    ndt.setTransformationEpsilon(ndt_transformation_epsilon_);//设置迭代结束的条件
    ndt.setStepSize(ndt_step_size_);//0.1改0.2没影响
    ndt.setResolution(ndt_resolution_);//0.2在停车场只有10cm柱子的时候比较好，0.5会出现匹配问题
    //ndt.setMaximumIterations (m_NdtMaximumIterations);//30改成5 没影响,耗时不变，都是提前跳出的
    ndt.setOulierRatio(ndt_oulier_ratio_);

    trans_map << 0.73042, 0.673141, 0.115616, 103.989,
        -0.659973, 0.739193, -0.13427, 3.71714,
        -0.175845, 0.0217698, 0.984177, 5.17327,
        0, 0, 0, 1;

    left_trans << 0.795087, 0.572453, -0.200335, 6.89183,
        -0.605218, 0.770305, -0.200853, 7.94558,
        0.0393396, 0.280942, 0.958918, 0.958656,
        0, 0, 0, 1;
  }
  void leftPointCallback(const sensor_msgs::PointCloud2ConstPtr &input) {
    //cout << "I RECEIVED INPUT !" << endl;
//    std::lock_guard<std::mutex> lock(cloud_lock_);
    pcl::fromROSMsg(*input, *left_cloud);
    left_cir_buf_cloud.push_back(*left_cloud);
  }
  void rightPointCallback(const sensor_msgs::PointCloud2ConstPtr &input) {
    //cout << "I RECEIVED INPUT !" << endl;
//    std::lock_guard<std::mutex> lock(cloud_lock_);
    pcl::fromROSMsg(*input, *right_cloud);
    right_cir_buf_cloud.push_back(*right_cloud);
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
  void transformCloud(Eigen::Matrix4f &transform);
  Eigen::Matrix4f getTransform();
  Eigen::Matrix4f getTransform(std::vector<float> ypr_params, std::vector<float> xyz_params);
  void pubTransformedCloud();
  bool loadSourceCloud();
  tf::Transform broadcasterTF();
  void cloudAlign(Eigen::Matrix4f &trans);
  void filterPointByPlane(std::vector<std::pair<int, float>> &back_dis_vec);
  float horizonPlane(pcl::PointCloud<pcl::PointXYZ> &point);
  void repairModel(vector<std::pair<int, float>> &dis_vec);
  void removeOutlier();
  void euclideanCluster();
  void pubROIMarker();
  void gridROI();
  void transSourceCloud(Eigen::Matrix4f &transform);
  void loadLidarData();
//  inline void setPtr() {
//    pcl::PointCloud<pcl::PointXYZ>::Ptr temp(new pcl::PointCloud<pcl::PointXYZ>);
//    if (!left_cir_buf_cloud.empty()) {
//      temp->clear();
//      for (auto it = left_cir_buf_cloud.begin(); it != left_cir_buf_cloud.end(); it++) {
//        *temp += *it;
//      }
//    }
//    *source_cloud = *temp;
//    if (!right_cir_buf_cloud.empty()) {
//      temp->clear();
//      for (auto it = right_cir_buf_cloud.begin(); it != right_cir_buf_cloud.end(); it++) {
//        *temp += *it;
//      }
//    }
//    *cloud_in = *temp;
//  }
  inline void addLeftRight() {
//    cloud_trans->clear();
    if (add_cloud) {
      pcl::PointCloud<pcl::PointXYZ>::Ptr temp_cloud(new pcl::PointCloud<pcl::PointXYZ>);
      //     Eigen::Matrix4f trans = this->getTransform(left_2_right_YPR_, left_2_right_xyz_);
      Eigen::Matrix4f trans = this->getTransform(lidar_ypr_params_, lidar_xyz_params_);
//      Eigen::Matrix4f trans_ = trans * left_trans ;
      pcl::transformPointCloud(*left_cloud, *temp_cloud, left_trans);
      cloud_in->clear();
      *cloud_in = *right_cloud + *temp_cloud;
      pcl::transformPointCloud(*cloud_in, *cloud_in, trans);
    } else {
      *cloud_in = *left_cloud;
      *source_cloud = *right_cloud;
    }
  }
  std::vector<float> ypr_params_;
  std::vector<float> xyz_params_;
  std::vector<float> lidar_ypr_params_;
  std::vector<float> lidar_xyz_params_;
  std::vector<float> left_2_right_YPR_;
  std::vector<float> left_2_right_xyz_;

  bool hand_adjust = false;

 private:
  ros::NodeHandle nh;
  ros::NodeHandle nh_private;
  ros::Publisher pub;
  ros::Publisher source_pub;
  ros::Publisher align_pub;
  ros::Publisher clustered_pub;
  ros::Publisher marker_pub;
  ros::Publisher filtered_pub;
  ros::Publisher cloud_in_pub;

  ros::Subscriber left_sub;
  ros::Subscriber right_sub;
  tf::TransformBroadcaster br;
  dynamic_reconfigure::Server<region_growing_segmentation::cloud_tutorialsConfig> server;
  dynamic_reconfigure::Server<region_growing_segmentation::cloud_tutorialsConfig>::CallbackType f;
  pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_in{new pcl::PointCloud<pcl::PointXYZ>};//转换为pcl格式
  pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_trans{new pcl::PointCloud<pcl::PointXYZ>};//加工后的pcl格式
  pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_aligned{new pcl::PointCloud<pcl::PointXYZ>};
  pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_filtered{new pcl::PointCloud<pcl::PointXYZ>};
  pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud_clustered{new pcl::PointCloud<pcl::PointXYZRGB>};
  pcl::PointCloud<pcl::PointXYZ>::Ptr source_cloud{new pcl::PointCloud<pcl::PointXYZ>};
  pcl::PointCloud<pcl::PointXYZ>::Ptr left_cloud{new pcl::PointCloud<pcl::PointXYZ>};
  pcl::PointCloud<pcl::PointXYZ>::Ptr right_cloud{new pcl::PointCloud<pcl::PointXYZ>};
  pcl::PointCloud<pcl::PointXYZ>::Ptr map_cloud{new pcl::PointCloud<pcl::PointXYZ>};
  boost::circular_buffer<pcl::PointCloud<pcl::PointXYZ>> left_cir_buf_cloud;
  boost::circular_buffer<pcl::PointCloud<pcl::PointXYZ>> right_cir_buf_cloud;

  sensor_msgs::PointCloud2 cloud_out;//转换为ros格式
  sensor_msgs::PointCloud2 source_cloud_out;
  sensor_msgs::PointCloud2 cloud_aligned_out;
  sensor_msgs::PointCloud2 cloud_clustered_out;
  sensor_msgs::PointCloud2 cloud_filtered_out;
  sensor_msgs::PointCloud2 cloud_in_out;

  float roll, pitch, yaw = 0;
  float x, y, z = 0;
  std::mutex cloud_lock_;
  std::mutex trans_lock_;

  pclomp::NormalDistributionsTransform<pcl::PointXYZ, pcl::PointXYZ> ndt;
  //Ndt parameters
  double ndt_transformation_epsilon_;
  double ndt_step_size_;
  double ndt_resolution_;
  int ndt_maximum_iterations_;
  double ndt_oulier_ratio_;

  std::vector<float> left_plane_params;
  std::vector<float> right_plane_params;
  std::vector<float> back_plane_params;

  std::vector<std::pair<int, pcl::PointXYZ>> min_x_point_cluster;
  std::vector<std::pair<int, pcl::PointXYZ>> max_x_point_cluster;
  std::vector<std::pair<int, pcl::PointXYZ>> min_y_point_cluster;
  std::vector<std::pair<int, pcl::PointXYZ>> max_y_point_cluster;
  Eigen::Vector4f min_pt, max_pt;

  std::vector<float> hight_vec;
  bool load_map = false;
  bool add_cloud = true;
  bool align_ = false;

  Eigen::Matrix4f trans_map;
  Eigen::Matrix4f left_trans;
};

Eigen::Matrix4f AdjustCloud::getTransform() {
  std::lock_guard<std::mutex> lock(trans_lock_);
  Eigen::Vector3f euler_angle(yaw, pitch, roll);
  Eigen::Vector3f trans_vector(x, y, z);
  Eigen::AngleAxisf rotation_vector;
  rotation_vector = Eigen::AngleAxisf(euler_angle[0], Eigen::Vector3f::UnitZ()) *
      Eigen::AngleAxisf(euler_angle[1], Eigen::Vector3f::UnitY()) *
      Eigen::AngleAxisf(euler_angle[2], Eigen::Vector3f::UnitX());
  Eigen::Isometry3f T = Eigen::Isometry3f::Identity();
  T.rotate(rotation_vector);
  T.pretranslate(trans_vector);
  Eigen::Matrix4f rotation_matrix = T.matrix();
  return rotation_matrix;
}
Eigen::Matrix4f AdjustCloud::getTransform(std::vector<float> ypr_params, std::vector<float> xyz_params) {
  Eigen::Vector3f euler_angle(ypr_params[0] * PI / 180, ypr_params[1] * PI / 180, ypr_params[2] * PI / 180);
  Eigen::Vector3f trans_vector(xyz_params[0], xyz_params[1], xyz_params[2]);
  Eigen::AngleAxisf rotation_vector;
  rotation_vector = Eigen::AngleAxisf(euler_angle[0], Eigen::Vector3f::UnitZ()) *
      Eigen::AngleAxisf(euler_angle[1], Eigen::Vector3f::UnitY()) *
      Eigen::AngleAxisf(euler_angle[2], Eigen::Vector3f::UnitX());
  Eigen::Isometry3f T = Eigen::Isometry3f::Identity();
  T.rotate(rotation_vector);
  T.pretranslate(trans_vector);
  Eigen::Matrix4f rotation_matrix = T.matrix();
  return rotation_matrix;
}

void AdjustCloud::transSourceCloud(Eigen::Matrix4f &transform) {
  if (!source_cloud->empty()) {
    Eigen::Matrix4f t_inverse;
    t_inverse.block(0, 0, 3, 3) = trans_map.block(0, 0, 3, 3).transpose();
    t_inverse.block(3, 0, 1, 3) = trans_map.block(3, 0, 1, 3);
    t_inverse.block(0, 3, 3, 1) = -t_inverse.block(0, 0, 3, 3) * trans_map.block(0, 3, 3, 1);
    t_inverse.block(3, 3, 1, 1) = trans_map.block(3, 3, 1, 1);
//    t_inverse.block(0, 0, 3, 3) = transform.block(0, 0, 3, 3).transpose();
//    t_inverse.block(3, 0, 1, 3) = transform.block(3, 0, 1, 3);
//    t_inverse.block(0, 3, 3, 1) = -t_inverse.block(0, 0, 3, 3) * transform.block(0, 3, 3, 1);
//    t_inverse.block(3, 3, 1, 1) = transform.block(3, 3, 1, 1);
    Eigen::Matrix4f rotation_matrix = this->getTransform(lidar_ypr_params_, lidar_xyz_params_);
    Eigen::Matrix4f tmp;
    tmp = rotation_matrix * t_inverse;
    pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_tmp{new pcl::PointCloud<pcl::PointXYZ>};//加工后的pcl格式

    pcl::transformPointCloud(*source_cloud, *cloud_trans, tmp);
//    pcl::transformPointCloud(*cloud_in, *cloud_tmp, rotation_matrix);
//    *cloud_in = *cloud_tmp;
  }
}
void AdjustCloud::transformCloud(Eigen::Matrix4f &transform) {
  Eigen::Isometry3f T = Eigen::Isometry3f::Identity();
  Eigen::Matrix4f matrix4f = T.matrix();
  if (!cloud_in->empty() && hand_adjust) {
    //std::lock_guard<std::mutex> lock(cloud_lock_);
//    Eigen::Matrix4f tmp;
//    tmp = trans_map * transform;
    cloud_trans->clear();
    pcl::transformPointCloud(*cloud_in, *cloud_trans, transform);
  }
}
void AdjustCloud::pubTransformedCloud() {
  //std::lock_guard<std::mutex> lock(cloud_lock_);
  ros::Time time = ros::Time::now();
  if (!cloud_trans->empty()) {
    pcl::toROSMsg(*cloud_trans, cloud_out);
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
  if (!cloud_aligned->empty()) {
    pcl::toROSMsg(*cloud_aligned, cloud_aligned_out);
    cloud_aligned_out.header.stamp = time;
    cloud_aligned_out.header.frame_id = "livox_frame";
    align_pub.publish(cloud_aligned_out);
  }
  if (!cloud_clustered->empty()) {
    pcl::toROSMsg(*cloud_clustered, cloud_clustered_out);
    cloud_clustered_out.header.stamp = time;
    cloud_clustered_out.header.frame_id = "livox_frame";
    clustered_pub.publish(cloud_clustered_out);
  }
//  if (!cloud_filtered->empty()) {
//    pcl::toROSMsg(*cloud_filtered, cloud_filtered_out);
//    cloud_filtered_out.header.stamp = time;
//    cloud_filtered_out.header.frame_id = "livox_frame";
//    filtered_pub.publish(cloud_filtered_out);
//  }
  if (!cloud_in->empty()) {
    pcl::toROSMsg(*cloud_in, cloud_in_out);
    cloud_in_out.header.stamp = time;
    cloud_in_out.header.frame_id = "livox_frame";
    cloud_in_pub.publish(cloud_in_out);
  }
}
bool AdjustCloud::loadSourceCloud() {
//  if (!pcl::io::loadPCDFile("/home/hjx/based_point_segment_ws/left_dense_pcd_2.pcd", *source_cloud)) {
//    std::cerr << "success to load " << "source_pcd" << std::endl;
//    // downsampling
//    pcl::PointCloud<pcl::PointXYZ>::Ptr downsampled(new pcl::PointCloud<pcl::PointXYZ>());
//    pcl::VoxelGrid<pcl::PointXYZ> voxelgrid;
//    voxelgrid.setLeafSize(0.5f, 0.5f, 0.5f);
//    voxelgrid.setInputCloud(source_cloud);
//    voxelgrid.filter(*downsampled);
//    *source_cloud = *downsampled;
//  }
//  if (!pcl::io::loadPCDFile("/home/hjx/based_point_segment_ws/right_dense.pcd", *cloud_in)) {
//    std::cerr << "success to load " << "add_pcd" << std::endl;
//    // downsampling
//    pcl::PointCloud<pcl::PointXYZ>::Ptr downsampled(new pcl::PointCloud<pcl::PointXYZ>());
//    pcl::VoxelGrid<pcl::PointXYZ> voxelgrid;
//    voxelgrid.setLeafSize(0.5f, 0.5f, 0.5f);
//    voxelgrid.setInputCloud(cloud_in);
//    voxelgrid.filter(*downsampled);
//    *cloud_in = *downsampled;
//  }
//  pcl::PointCloud<pcl::PointXYZ>::Ptr temp(new pcl::PointCloud<pcl::PointXYZ>);
//  if (!left_cir_buf_cloud.empty()) {
//    temp->clear();
//    for (auto it = left_cir_buf_cloud.begin(); it != left_cir_buf_cloud.end(); it++) {
//      *temp += *it;
//    }
//  }
//  *left_cloud = *temp;
//  temp->clear();
//  if (!right_cir_buf_cloud.empty()) {
//    for (auto it = right_cir_buf_cloud.begin(); it != right_cir_buf_cloud.end(); it++) {
//      *temp += *it;
//    }
//  }
//  *right_cloud = *temp;
//  pcl::PointCloud<pcl::PointXYZ>::Ptr downsampled(new pcl::PointCloud<pcl::PointXYZ>());
//  pcl::VoxelGrid<pcl::PointXYZ> voxelgrid;
//  voxelgrid.setLeafSize(0.5f, 0.5f, 0.5f);
//  voxelgrid.setInputCloud(left_cloud);
//  voxelgrid.filter(*downsampled);
//  *left_cloud = *downsampled;
//
//  voxelgrid.setInputCloud(right_cloud);
//  voxelgrid.filter(*downsampled);
//  *right_cloud = *downsampled;

  if (load_map) {
    if (!pcl::io::loadPCDFile("/home/hjx/based_point_segment_ws/sub_map.pcd", *map_cloud)) {
      std::cerr << "success to load " << "source_pcd" << std::endl;
      // downsampling
      pcl::PointCloud<pcl::PointXYZ>::Ptr downsampled(new pcl::PointCloud<pcl::PointXYZ>());
      pcl::VoxelGrid<pcl::PointXYZ> voxelgrid;
      voxelgrid.setLeafSize(0.5f, 0.5f, 0.5f);
      voxelgrid.setInputCloud(map_cloud);
      voxelgrid.filter(*downsampled);
      *map_cloud = *downsampled;
      *source_cloud = *map_cloud;
    }
  }
//  if (!pcl::io::loadPCDFile("/home/hjx/based_point_segment_ws/add.pcd", *cloud_in)) {
//    std::cerr << "success to load " << "add_pcd" << std::endl;
//    // downsampling
//    pcl::PointCloud<pcl::PointXYZ>::Ptr downsampled(new pcl::PointCloud<pcl::PointXYZ>());
//    pcl::VoxelGrid<pcl::PointXYZ> voxelgrid;
//    voxelgrid.setLeafSize(0.1f, 0.1f, 0.1f);
//    voxelgrid.setInputCloud(cloud_in);
//    voxelgrid.filter(*downsampled);
//    *cloud_in = *downsampled;
//  }
}
tf::Transform AdjustCloud::broadcasterTF() {
  tf::Transform transform;
  tf::Vector3 vector3(x, y, z);
  transform.setOrigin(vector3);
  tf::Quaternion q;
  q.setRPY(roll, pitch, yaw);
  transform.setRotation(q);
  this->br.sendTransform(tf::StampedTransform(transform, ros::Time::now(), "livox_frame", "target_frame"));
  return transform;
}
void AdjustCloud::cloudAlign(Eigen::Matrix4f &trans) {
  if (!source_cloud->empty() && !cloud_in->empty() && align_) {
    //std::lock_guard<std::mutex> lock(cloud_lock_);
    cloud_in->is_dense = false;
    std::vector<int> out_inliers;
    pcl::PointCloud<pcl::PointXYZ>::Ptr final(new pcl::PointCloud<pcl::PointXYZ>);
    pcl::removeNaNFromPointCloud(*cloud_in, *final, out_inliers);

    ndt.setInputTarget(source_cloud);
    ndt.setInputSource(final);
    ndt.align(*cloud_aligned, trans);
    cout << ndt.getFinalTransformation() << endl;

    //cout << "aligned score" << ndt.getFitnessScore() << endl;
    cout << "aligned over" << endl;
  }
}
void AdjustCloud::filterPointByPlane(std::vector<std::pair<int, float>> &back_dis_vec) {
  if (!cloud_in->empty()) {
    float horizon_dis, left_dis, right_dis, back_dis = 0;
    pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_temp{new pcl::PointCloud<pcl::PointXYZ>};
    for (int i = 0; i < cloud_in->size(); i++) {
      if (cloud_in->points[i].z < 0.3 || cloud_in->points[i].z > 3.5)
        continue;
//      if (cloud_in->points[i].y > 3.0)
//        continue;
//      horizon_dis = -0.271748 * cloud_in->points[i].x + 0.0127325 * cloud_in->points[i].y +
//          0.962284 * cloud_in->points[i].z + 10.0278;
      left_dis = left_plane_params[0] * cloud_in->points[i].x + left_plane_params[1] * cloud_in->points[i].y +
          left_plane_params[2] * cloud_in->points[i].z + left_plane_params[3];
      right_dis = right_plane_params[0] * cloud_in->points[i].x + right_plane_params[1] * cloud_in->points[i].y +
          right_plane_params[2] * cloud_in->points[i].z + right_plane_params[3];
      back_dis = back_plane_params[0] * cloud_in->points[i].x + back_plane_params[1] * cloud_in->points[i].y +
          back_plane_params[2] * cloud_in->points[i].z + back_plane_params[3];
//      cout << "horizon_dis " << horizon_dis << "left_dist " << left_dis << "right_dis " << right_dis << "back_dis "
//           << back_dis << endl;
      if (left_dis - 1 > 0 && right_dis + 0.5 < 0 && back_dis + 0.7 < 0) {
        cloud_temp->points.push_back(cloud_in->points[i]);
        back_dis_vec.push_back(std::make_pair(i, abs(back_dis)));
      }
    }
    *cloud_filtered = *cloud_temp;
    cout << "cloud_temp size " << cloud_temp->size() << endl;
  }
}
void AdjustCloud::repairModel(vector<std::pair<int, float>> &dis_vec) {
  if (!dis_vec.empty()) {
    float repaired_dis_y = 0;
    float repaired_dis_z = 0;
    int j = 0;
    pcl::PointCloud<pcl::PointXYZ>::Ptr new_point(new pcl::PointCloud<pcl::PointXYZ>);
    for (int i = 0; i < dis_vec.size(); i++) {
      //cout << "dis_vec" << dis_vec[i] << endl;
      if (dis_vec[i].second < 5) {
        new_point->points.push_back(cloud_filtered->points[i]);
        repaired_dis_y =
            (0.328414 * cloud_filtered->points[i].x + 0.0246555 * cloud_filtered->points[i].z - 63.1203) / 0.944212;;
        repaired_dis_z =
            (-0.00541356 * cloud_filtered->points[i].x + 0.0257402 * cloud_filtered->points[i].y + 5.28737) / -0.999654
                + 0.3;
        //cloud_filtered->points[i].y = repaired_dis_y;
        source_cloud->points[dis_vec[i].first].z = repaired_dis_z;
        new_point->points[j].y = repaired_dis_y;
        j++;
      } else {
        repaired_dis_z =
            (-0.00541356 * cloud_filtered->points[i].x + 0.0257402 * cloud_filtered->points[i].y + 5.28737) / -0.999654
                + 0.3;
        source_cloud->points[dis_vec[i].first].z = repaired_dis_z;
      }
    }
    *source_cloud += *new_point;
    cout << source_cloud->size() << endl;
  }
}
void AdjustCloud::removeOutlier() {
  if (!cloud_filtered->empty()) {
    pcl::StatisticalOutlierRemoval<pcl::PointXYZ> sor;
    pcl::PointCloud<pcl::PointXYZ>::Ptr filter(new pcl::PointCloud<pcl::PointXYZ>);
    sor.setInputCloud(cloud_filtered);
    //cloud_filtered->clear();
    sor.setMeanK(50);
    sor.setStddevMulThresh(2);
    sor.filter(*filter);
    *cloud_filtered = *filter;
  }
}
void AdjustCloud::euclideanCluster() {

  pcl::getMinMax3D(*cloud_filtered, min_pt, max_pt);

//  pcl::search::KdTree<pcl::PointXYZ>::Ptr tree(new pcl::search::KdTree<pcl::PointXYZ>);
//  tree->setInputCloud(cloud_filtered);
//  std::vector<pcl::PointIndices> cluster_indices;
//  pcl::EuclideanClusterExtraction<pcl::PointXYZ> ec;
//  ec.setClusterTolerance(1); //设置近邻搜索的搜索半径为2cm
//  ec.setMinClusterSize(100);//设置一个聚类需要的最少点数目为100
//  ec.setMaxClusterSize(25000); //设置一个聚类需要的最大点数目为25000
//  ec.setSearchMethod(tree);//设置点云的搜索机制
//  ec.setInputCloud(cloud_filtered);
//  ec.extract(cluster_indices);//从点云中提取聚类，并将点云索引保存在cluster_indices中
//  int color_bar[][3] =
//      {
//          {255, 0, 0},
//          {0, 255, 0},
//          {0, 0, 255},
//          {0, 255, 255},
//          {255, 255, 0},
//          {255, 255, 255},
//          {255, 0, 255}
//      };
//  int k = 0;
//  int j = 0;
//  cloud_clustered->points.resize(cloud_filtered->size());
//
//  for (std::vector<pcl::PointIndices>::const_iterator it = cluster_indices.begin(); it != cluster_indices.end(); it++) {
//    float min_x = std::numeric_limits<float>::max();
//    float min_y = std::numeric_limits<float>::max();
//    float max_x = -std::numeric_limits<float>::max();
//    float max_y = -std::numeric_limits<float>::max();
//    int min_x_index, max_x_index, min_y_index, max_y_index;
//    for (std::vector<int>::const_iterator pit = it->indices.begin(); pit != it->indices.end(); pit++) {
//      {
//        cloud_clustered->points[j].x = cloud_filtered->points[*pit].x;
//        cloud_clustered->points[j].y = cloud_filtered->points[*pit].y;
//        cloud_clustered->points[j].z = cloud_filtered->points[*pit].z;
//        cloud_clustered->points[j].r = color_bar[k][0];
//        cloud_clustered->points[j].g = color_bar[k][1];
//        cloud_clustered->points[j].b = color_bar[k][2];
//        j++;
//      }
//      {
//        if (cloud_filtered->points[*pit].x < min_x) {
//          min_x = cloud_filtered->points[*pit].x;
//          min_x_index = *pit;
//        }
//        if (cloud_filtered->points[*pit].y < min_y) {
//          min_y = cloud_filtered->points[*pit].y;
//          min_y_index = *pit;
//        }
//        if (cloud_filtered->points[*pit].x > max_x) {
//          max_x = cloud_filtered->points[*pit].x;
//          max_x_index = *pit;
//        }
//        if (cloud_filtered->points[*pit].y > max_y) {
//          max_y = cloud_filtered->points[*pit].y;
//          max_y_index = *pit;
//        }
//      }
//    }
//    {
//      min_x_point_cluster.push_back(std::make_pair(k, cloud_filtered->points[min_x_index]));
//      max_x_point_cluster.push_back(std::make_pair(k, cloud_filtered->points[max_x_index]));
//      min_y_point_cluster.push_back(std::make_pair(k, cloud_filtered->points[min_y_index]));
//      max_y_point_cluster.push_back(std::make_pair(k, cloud_filtered->points[max_y_index]));
//    }
//    k++;
//    if (k > 6) k = 0;
//  }
}
void AdjustCloud::pubROIMarker() {
  if (1) {
    visualization_msgs::Marker line_points, line_list;
    line_list.header.frame_id = "livox_frame";
    line_list.header.stamp = ros::Time::now();
    line_list.ns = "lines";
    line_list.action = visualization_msgs::Marker::ADD;
    line_list.pose.orientation.w = 1.0;
    line_list.id = 0;
    line_list.type = visualization_msgs::Marker::LINE_LIST;
    line_list.scale.x = 0.01;
    line_list.color.b = 1.0;
    line_list.color.a = 1.0;
    geometry_msgs::Point point_ll, point_lu, point_rl, point_ru;
    geometry_msgs::Point point_1, point_2;

    point_ll.x = min_pt.x();
    point_ll.y = min_pt.y();
    point_ll.z = 0;

    point_lu.x = min_pt.x();
    point_lu.y = max_pt.y();
    point_lu.z = 0;

    point_ru.x = max_pt.x();
    point_ru.y = max_pt.y();
    point_ru.z = 0;

    point_rl.x = max_pt.x();
    point_rl.y = min_pt.y();
    point_rl.z = 0;

    line_list.points.push_back(point_ll);
    line_list.points.push_back(point_lu);
    line_list.points.push_back(point_lu);
    line_list.points.push_back(point_ru);
    line_list.points.push_back(point_ru);
    line_list.points.push_back(point_rl);
    line_list.points.push_back(point_rl);
    line_list.points.push_back(point_ll);
    for (float x_1 = min_pt.x(); x_1 <= max_pt.x();) {
      point_1.x = x_1;
      point_1.y = min_pt.y();
      point_1.z = 0;

      point_2.x = x_1;
      point_2.y = max_pt.y();
      point_2.z = 0;

      x_1 += 0.1;
      line_list.points.push_back(point_1);
      line_list.points.push_back(point_2);
    }
    for (float y_1 = min_pt.y(); y_1 <= max_pt.y();) {
      point_1.x = min_pt.x();
      point_1.y = y_1;
      point_1.z = 0;

      point_2.x = max_pt.x();
      point_2.y = y_1;
      point_2.z = 0;

      y_1 += 0.1;
      line_list.points.push_back(point_1);
      line_list.points.push_back(point_2);
    }
    marker_pub.publish(line_list);
//    float z =
//        (-0.00541356 * min_x_point_cluster[0].second.x + 0.0257402 * min_x_point_cluster[0].second.y + 5.28737)
//            / -0.999654
//            + 0.3;
//    geometry_msgs::Point point_ll, point_lu, point_lr, point_ur;
//    geometry_msgs::Point point_1, point_2;
//    for (int i = 0; i < min_x_point_cluster.size(); i++) {
//      point_ll.x = min_x_point_cluster[i].second.x;
//      point_ll.y = min_y_point_cluster[i].second.y;
//      point_ll.z = z;
//      point_lu.x = min_x_point_cluster[i].second.x;
//      point_lu.y = max_y_point_cluster[i].second.y;
//      point_lu.z = z;
//      point_lr.x = max_x_point_cluster[i].second.x;
//      point_lr.y = min_y_point_cluster[i].second.y;
//      point_lr.z = z;
//      point_ur.x = max_x_point_cluster[i].second.x;
//      point_ur.y = max_y_point_cluster[i].second.y;
//      point_ur.z = z;
//      line_list.points.push_back(point_ll);
//      line_list.points.push_back(point_lu);
//      line_list.points.push_back(point_lu);
//      line_list.points.push_back(point_ur);
//      line_list.points.push_back(point_ur);
//      line_list.points.push_back(point_lr);
//      line_list.points.push_back(point_lr);
//      line_list.points.push_back(point_ll);
//
//      for (float x_1 = min_x_point_cluster[i].second.x; x_1 <= max_x_point_cluster[i].second.x;) {
//        point_1.x = x_1;
//        point_1.y = min_y_point_cluster[i].second.y;
//        point_1.z = z;
//
//        point_2.x = x_1;
//        point_2.y = max_y_point_cluster[i].second.y;
//        point_2.z = z;
//
//        x_1 += 0.1;
//        line_list.points.push_back(point_1);
//        line_list.points.push_back(point_2);
//      }
//      for (float y_1 = min_y_point_cluster[i].second.y; y_1 <= max_y_point_cluster[i].second.y;) {
//        point_1.x = min_x_point_cluster[i].second.x;
//        point_1.y = y_1;
//        point_1.z = z;
//
//        point_2.x = max_x_point_cluster[i].second.x;
//        point_2.y = y_1;
//        point_2.z = z;
//
//        y_1 += 0.1;
//        line_list.points.push_back(point_1);
//        line_list.points.push_back(point_2);
//      }
//      marker_pub.publish(line_list);
//    }
  }
}
void AdjustCloud::gridROI() {
  if (max_pt.x() - min_pt.x() > 0 && max_pt.y() - min_pt.y() > 0) {
    float x_next_step, y_next_step;
    float average_hight, total_hight;
    int num;
    int cols = (int) ((max_pt.y() - min_pt.y()) / 0.1) + 1;
    int rows = (int) ((max_pt.x() - min_pt.x()) / 0.1) + 1;
//    cout << "cols " << cols << " rows " << rows << endl;
    cv::Mat proj_mat_1(rows, cols, CV_32F, cv::Scalar::all(0));
    cv::Mat proj_mat_2(rows, cols, CV_32F, cv::Scalar::all(0));
    cv::Mat proj_num(rows, cols, CV_32F, cv::Scalar::all(0.0));

//    int i, j = 0;
//    pcl::PointCloud<pcl::PointXYZ>::Ptr copy_cloud(new pcl::PointCloud<pcl::PointXYZ>);
//    pcl::copyPointCloud(*cloud_filtered, *copy_cloud);

    std::vector<Point2Pixel> total_pixel;
    Point2Pixel temp_pt;
//    int x_row=0,y_col=0;
    for (auto it = cloud_filtered->points.begin(); it != cloud_filtered->points.end(); it++) {
      temp_pt.x = (int) ((it->x - min_pt.x()) / 0.1);
      temp_pt.y = (int) ((it->y - min_pt.y()) / 0.1);
      temp_pt.hight = it->z;
      total_pixel.push_back(temp_pt);
    }
    for (auto pixel = total_pixel.begin(); pixel != total_pixel.end(); pixel++) {
      proj_mat_1.at<float>(pixel->x, pixel->y) += pixel->hight;
      proj_num.at<float>(pixel->x, pixel->y) += 1.0;
    }

//    for (float x_start = min_pt.x(); x_start <= max_pt.x(); x_start += 0.1, i++) {
//      x_next_step = x_start + 0.1;
//      j = 0;
//      for (float y_start = min_pt.y(); y_start <= max_pt.y(); y_start += 0.1, j++) {
//        y_next_step = y_start + 0.1;
//        for (auto it = cloud_filtered->points.begin(); it != cloud_filtered->points.end();) {
////          if (it->x == 0 && it->y == 0 && it->z == 0) break;
////          if (it->y > y_start && it->y < y_next_step) {
//          if (it->y >= y_start && it->y <= y_next_step && it->x >= x_start && it->x <= x_next_step) {
//            total_hight += it->z;
//            num += 1;
//            it = cloud_filtered->erase(it);
//          } else {
//            it++;
//          }
////          }
//        }
//        if (num != 0) {
//          average_hight = total_hight / num;
//          proj_mat_2.at<float>(i, j) = average_hight;
//          total_hight = 0;
//          num = 0;
//          average_hight = 0;
//        }
//      }
//    }
    cv::Mat image_1(proj_mat_1.rows, proj_mat_1.cols, CV_32F, cv::Scalar::all(0));
    cv::Mat image_2(proj_mat_1.rows, proj_mat_1.cols, CV_32F, cv::Scalar::all(0));
//    image_1 = proj_mat_2 * 10.0;
//    image_2 = proj_mat_1 / proj_num;
    cv::divide(proj_mat_1, proj_num, image_1);
    cv::Mat element = cv::getStructuringElement(MORPH_RECT, cv::Size(5, 5));
    cv::dilate(image_1, image_1, element);
    bool skip_low = true;
    bool update_edge = true;
    int peak_row = 0;
    int edge_row = 0;
    float peak_row_data = 0;
    for (int image_col = 0; image_col < image_1.cols; image_col++) {
      skip_low = true;
      update_edge = true;
      peak_row = 0;
      peak_row_data = 0;
      for (int image_row = 0; image_row < image_1.rows; image_row++) {
        if (image_1.at<float>(image_row, image_col) == 0 && skip_low) continue;
        else {
          skip_low = false;
          if (image_1.at<float>(image_row, image_col) > peak_row_data) {
            peak_row_data = image_1.at<float>(image_row, image_col);
            peak_row = image_row;
            if (update_edge) {
              edge_row = image_row;
              update_edge = false;
            }
          } else if (image_1.at<float>(image_row, image_col) == 0) {
            peak_row_data = 0;
            update_edge = true;
            if (2 * peak_row - image_row >= 0 && image_row - peak_row <= peak_row - edge_row) {
              image_1.at<float>(image_row, image_col) = image_1.at<float>((2 * peak_row - image_row), image_col);
            } else {
              image_1.at<float>(image_row, image_col) = 0;
            }
          }
        }
      }
    }
//    cv::dilate(image_1, image_1, element);
//    std::string filename = "/home/hjx/based_point_segment_ws/proj_image_2.csv";
//    ofstream myfile;
//    myfile.open(filename);
//    for (int i = 0; i < image_1.rows; i++) {
//      myfile << cv::format(image_1.row(i), cv::Formatter::FMT_CSV) << std::endl;
//    }
//    myfile << cv::format(image_1.row(104),cv::Formatter::FMT_CSV) << std::endl;
//    myfile.close();
//    myfile << cv::format(image_1, cv::Formatter::FMT_CSV) << std::endl;

//    image_1 = image_1 * 10.0;  //Expand the gray value for easy visualization
//    cout << image_1.row(104) << endl;
//    cv::imwrite("/home/hjx/based_point_segment_ws/proj_image.jpg", image_1);
//    cout << cloud_filtered->size() << endl;
//    cout << copy_cloud->size() << endl;

//    for (int i = 0; i < min_x_point_cluster.size(); i++) {
//      cout << "col size " << (int) ((max_x_point_cluster[i].second.x - min_x_point_cluster[i].second.x) / 0.1) + 1
//           << endl;
//      cout << "row size " << (int) ((max_y_point_cluster[i].second.y - min_y_point_cluster[i].second.y) / 0.1) + 1
//           << endl;
//      x_start = min_x_point_cluster[i].second.x;
//      average_hight = 0;
//      num = 0;
//      for (; x_start < max_x_point_cluster[i].second.x; x_start += 0.1) {
//        y_start = min_y_point_cluster[i].second.y;
//        x_next_step = x_start + 0.1;
//        for (; y_start < max_y_point_cluster[i].second.y; y_start += 0.1) {
//          y_next_step = y_start + 0.1;
//          for (auto it = cloud_filtered->points.begin(); it != cloud_filtered->points.end(); it++) {
//            if (it->x == 0 && it->y == 0 && it->z == 0) break;
//            if (it->x > x_start && it->x < x_next_step) {
//              if (it->y > y_start && it->y < y_next_step) {
//                total_hight += it->z - z;
//                num += 1;
//              }
//            }
//          }
//          if (num != 0) {
//            average_hight = total_hight / num;
//            hight_vec.push_back(average_hight);
//            num = 0;
//            total_hight = 0.0;
//            average_hight = 0.0;
//          } else {
//            hight_vec.push_back(average_hight);
//          }
//        }
//      }
//    }

//    *cloud_filtered = * copy_cloud;
//    cout << "copy_cloud size " << copy_cloud->size() << endl;
  }
}
void AdjustCloud::loadLidarData() {
  pcl::PointCloud<pcl::PointXYZ>::Ptr temp(new pcl::PointCloud<pcl::PointXYZ>);
  if (!left_cir_buf_cloud.empty()) {
    temp->clear();
    for (auto it = left_cir_buf_cloud.begin(); it != left_cir_buf_cloud.end(); it++) {
      *temp += *it;
    }
  }
  *left_cloud = *temp;
  temp->clear();
  if (!right_cir_buf_cloud.empty()) {
    for (auto it = right_cir_buf_cloud.begin(); it != right_cir_buf_cloud.end(); it++) {
      *temp += *it;
    }
  }
  *right_cloud = *temp;
  pcl::PointCloud<pcl::PointXYZ>::Ptr downsampled(new pcl::PointCloud<pcl::PointXYZ>());
  pcl::VoxelGrid<pcl::PointXYZ> voxelgrid;
  voxelgrid.setLeafSize(0.5f, 0.5f, 0.5f);
  voxelgrid.setInputCloud(left_cloud);
  voxelgrid.filter(*downsampled);
  *left_cloud = *downsampled;

  voxelgrid.setInputCloud(right_cloud);
  voxelgrid.filter(*downsampled);
  *right_cloud = *downsampled;
}

int
main(int argc, char **argv) {
  ros::init(argc, argv, "AdjustCloud");
  AdjustCloud trans_pointcloud;
  ros::Rate sleep_rate(5);
  trans_pointcloud.loadSourceCloud();
  Eigen::Matrix4f rotation_matrix =
      trans_pointcloud.getTransform(trans_pointcloud.ypr_params_, trans_pointcloud.xyz_params_);
  trans_pointcloud.transSourceCloud(rotation_matrix);
  vector<std::pair<int, float>> dis_vec;
  trans_pointcloud.filterPointByPlane(dis_vec);
  trans_pointcloud.removeOutlier();
  trans_pointcloud.euclideanCluster();
//  trans_pointcloud.gridROI();
  while (ros::ok()) {
    //trans_pointcloud.broadcasterTF();
//    trans_pointcloud.setPtr();
    trans_pointcloud.loadLidarData();
    trans_pointcloud.addLeftRight();
    Eigen::Matrix4f transform = trans_pointcloud.getTransform();
//    if (trans_pointcloud.hand_adjust) {
    trans_pointcloud.transformCloud(transform);
//    } else {
//      trans_pointcloud.transformCloud(rotation_matrix);
//    }
    trans_pointcloud.cloudAlign(transform);
//    trans_pointcloud.pubROIMarker();
    trans_pointcloud.pubTransformedCloud();
    sleep_rate.sleep();
    ros::spinOnce();
  }
}
