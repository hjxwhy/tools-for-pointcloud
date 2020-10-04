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
#include <pcl/filters/statistical_outlier_removal.h>
#include <pcl_conversions/pcl_conversions.h>
#include <pcl/segmentation/sac_segmentation.h>
#include <pcl/common/transforms.h>
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
#include <math.h>
#include <dynamic_reconfigure/server.h>
#include "region_growing_segmentation/cloud_tutorialsConfig.h"
#include "pclomp/ndt_omp.h"
#define PI 3.14
using namespace std;
class AdjustCloud {
 public:
  AdjustCloud(pcl::PointCloud<pcl::PointXYZ>::Ptr source)
      : source_cloud(source), roll(0), pitch(0), yaw(-43.2), x(52.0), y(28.0), z(-1.0) {
    pub = nh.advertise<sensor_msgs::PointCloud2>("/trans_pointcloud", 1);
    source_pub = nh.advertise<sensor_msgs::PointCloud2>("/source_pointcloud", 1);
    align_pub = nh.advertise<sensor_msgs::PointCloud2>("/aligned_pointcloud", 1);
    clustered_pub = nh.advertise<sensor_msgs::PointCloud2>("/clustered_pointcloud", 1);
    filtered_pub = nh.advertise<sensor_msgs::PointCloud2>("/filtered_pointcloud", 1);
    marker_pub = nh.advertise<visualization_msgs::Marker>("visualization_marker", 1);

    sub = nh.subscribe("/livox/lidar", 1, &AdjustCloud::pointCallback, this);
    f = boost::bind(&AdjustCloud::callback, this, _1, _2);
    server.setCallback(f);

    nh_private = ros::NodeHandle("~");
    nh_private.param<double>("NDT_TransformationEpsilon", ndt_transformation_epsilon_, 0.001);
    nh_private.param<double>("NDT_StepSize", ndt_step_size_, 0.1);
    nh_private.param<double>("NDT_Resolution", ndt_resolution_, 1);
    nh_private.param<int>("NDT_MaximumIterations", ndt_maximum_iterations_, 30);
    nh_private.param<double>("NDT_OulierRatio", ndt_oulier_ratio_, 0.2);
    ndt.setMaximumIterations(ndt_maximum_iterations_);
    ndt.setNumThreads(omp_get_num_threads());
    ndt.setNeighborhoodSearchMethod(pclomp::DIRECT7);
    ndt.setTransformationEpsilon(ndt_transformation_epsilon_);//设置迭代结束的条件
    ndt.setStepSize(ndt_step_size_);//0.1改0.2没影响
    ndt.setResolution(ndt_resolution_);//0.2在停车场只有10cm柱子的时候比较好，0.5会出现匹配问题
    //ndt.setMaximumIterations (m_NdtMaximumIterations);//30改成5 没影响,耗时不变，都是提前跳出的
    ndt.setOulierRatio(ndt_oulier_ratio_);

    transtorm2 << 0.999821,   0.0153674,   0.0110796,    0.940724,
        -0.0154724,    0.999836,  0.00944811,     2.29541,
        -0.0109326, -0.00561784,    0.999894,     1.12445,
        0,           0,           0 ,          1;
  }
  void pointCallback(const sensor_msgs::PointCloud2ConstPtr &input) {
    //cout << "I RECEIVED INPUT !" << endl;
    //std::lock_guard<std::mutex> lock(cloud_lock_);
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
  void cloudAlign();
  void filterPointByPlane(std::vector<std::pair<int, float>> &back_dis_vec);
  float horizonPlane(pcl::PointCloud<pcl::PointXYZ> &point);
  void repairModel(vector<std::pair<int, float>> &dis_vec);
  void removeOutlier();
  void euclideanCluster();
  void pubROIMarker();
  void gridROI();
 private:
  ros::NodeHandle nh;
  ros::NodeHandle nh_private;
  ros::Publisher pub;
  ros::Publisher source_pub;
  ros::Publisher align_pub;
  ros::Publisher clustered_pub;
  ros::Publisher marker_pub;
  ros::Publisher filtered_pub;

  ros::Subscriber sub;
  tf::TransformBroadcaster br;
  dynamic_reconfigure::Server<region_growing_segmentation::cloud_tutorialsConfig> server;
  dynamic_reconfigure::Server<region_growing_segmentation::cloud_tutorialsConfig>::CallbackType f;
  pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_in{new pcl::PointCloud<pcl::PointXYZ>};//转换为pcl格式
  pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_trans{new pcl::PointCloud<pcl::PointXYZ>};//加工后的pcl格式
  pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_aligned{new pcl::PointCloud<pcl::PointXYZ>};
  pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_filtered{new pcl::PointCloud<pcl::PointXYZ>};
  pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud_clustered{new pcl::PointCloud<pcl::PointXYZRGB>};

  sensor_msgs::PointCloud2 cloud_out;//转换为ros格式
  sensor_msgs::PointCloud2 source_cloud_out;
  sensor_msgs::PointCloud2 cloud_aligned_out;
  sensor_msgs::PointCloud2 cloud_clustered_out;
  sensor_msgs::PointCloud2 cloud_filtered_out;

  float roll, pitch, yaw = 0;
  float x, y, z = 0;
  std::mutex cloud_lock_;
  std::mutex trans_lock_;
  pcl::PointCloud<pcl::PointXYZ>::Ptr source_cloud;

  pclomp::NormalDistributionsTransform<pcl::PointXYZ, pcl::PointXYZ> ndt;
  //Ndt parameters
  double ndt_transformation_epsilon_;
  double ndt_step_size_;
  double ndt_resolution_;
  int ndt_maximum_iterations_;
  double ndt_oulier_ratio_;

  std::vector<std::pair<int, pcl::PointXYZ>> min_x_point_cluster;
  std::vector<std::pair<int, pcl::PointXYZ>> max_x_point_cluster;
  std::vector<std::pair<int, pcl::PointXYZ>> min_y_point_cluster;
  std::vector<std::pair<int, pcl::PointXYZ>> max_y_point_cluster;

  std::vector<float> hight_vec;

  Eigen::Matrix4d transtorm2;
};

Eigen::Matrix4d AdjustCloud::getTransform() {
  std::lock_guard<std::mutex> lock(trans_lock_);
  Eigen::Vector3d euler_angle(yaw, pitch, roll);
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
    //std::lock_guard<std::mutex> lock(cloud_lock_);
    Eigen::Matrix4d tmp;
    tmp = transtorm2  * transform;
    cloud_trans->clear();
    pcl::transformPointCloud(*cloud_in, *cloud_trans, tmp);
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
  if (!cloud_filtered->empty()) {
    pcl::toROSMsg(*cloud_filtered, cloud_filtered_out);
    cloud_filtered_out.header.stamp = time;
    cloud_filtered_out.header.frame_id = "livox_frame";
    filtered_pub.publish(cloud_filtered_out);
  }
}
bool AdjustCloud::loadSourceCloud() {
  if (!pcl::io::loadPCDFile("/home/hjx/Documents/sub_map.pcd", *source_cloud)) {
    std::cerr << "success to load " << "source_pcd" << std::endl;
    // downsampling
    pcl::PointCloud<pcl::PointXYZ>::Ptr downsampled(new pcl::PointCloud<pcl::PointXYZ>());
    pcl::VoxelGrid<pcl::PointXYZ> voxelgrid;
    voxelgrid.setLeafSize(0.2f, 0.2f, 0.2f);
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
  this->br.sendTransform(tf::StampedTransform(transform, ros::Time::now(), "livox_frame", "target_frame"));
  return transform;
}
void AdjustCloud::cloudAlign() {
  if (!source_cloud->empty() && !cloud_trans->empty()) {
    //std::lock_guard<std::mutex> lock(cloud_lock_);
    cloud_trans->is_dense = false;
    std::vector<int> out_inliers;
    pcl::PointCloud<pcl::PointXYZ>::Ptr final(new pcl::PointCloud<pcl::PointXYZ>);
    pcl::removeNaNFromPointCloud(*cloud_trans, *final, out_inliers);
//
//    pcl::PointCloud<pcl::PointXYZ>::Ptr final_ds(new pcl::PointCloud<pcl::PointXYZ>);
//    pcl::VoxelGrid<pcl::PointXYZ> voxelgrid;
//    voxelgrid.setLeafSize(0.5f, 0.5f, 0.5f);
//    voxelgrid.setInputCloud(final);
//    voxelgrid.filter(*final_ds);

    ndt.setInputTarget(source_cloud);
    ndt.setInputSource(final);
    ndt.align(*cloud_aligned);
    cout << ndt.getFinalTransformation() << endl;
    cout << "aligned over" << endl;
  }
}
void AdjustCloud::filterPointByPlane(std::vector<std::pair<int, float>> &back_dis_vec) {
  if (!source_cloud->empty()) {
    float horizon_dis, left_dis, right_dis, back_dis = 0;
    pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_temp{new pcl::PointCloud<pcl::PointXYZ>};
    for (int i = 0; i < source_cloud->size(); i++) {
      if (source_cloud->points[i].z > 0.5)
        continue;
      if (source_cloud->points[i].y > 3.0)
        continue;
      horizon_dis = -0.00541356 * source_cloud->points[i].x + 0.0257402 * source_cloud->points[i].y +
          0.999654 * source_cloud->points[i].z + 5.28737;
      left_dis = 0.99537 * source_cloud->points[i].x + 0.0945941 * source_cloud->points[i].y +
          0.017061 * source_cloud->points[i].z - 115.25;
      right_dis = 0.993879 * source_cloud->points[i].x + 0.106574 * source_cloud->points[i].y +
          0.0290756 * source_cloud->points[i].z - 103.267;
      back_dis = 0.328414 * source_cloud->points[i].x - 0.944212 * source_cloud->points[i].y +
          0.0246555 * source_cloud->points[i].z - 63.1203;
      if (horizon_dis - 0.5 > 0 && left_dis + 0.5 < 0 && right_dis - 0.5 > 0 && back_dis + 0.5 < 0) {
        cloud_temp->points.push_back(source_cloud->points[i]);
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
//        new_point->points.push_back(cloud_filtered->points[i]);
//        repaired_dis_y =
//            (0.328414 * cloud_filtered->points[i].x + 0.0246555 * cloud_filtered->points[i].z - 63.1203) / 0.944212;;
//        repaired_dis_z =
//            (-0.00541356 * cloud_filtered->points[i].x + 0.0257402 * cloud_filtered->points[i].y + 5.28737) / -0.999654 + 0.3;
//        //cloud_filtered->points[i].y = repaired_dis_y;
//        cloud_filtered->points[i].z = repaired_dis_z;
//        new_point->points[j].y = repaired_dis_y;
//        j++;
//      } else {
//        repaired_dis_z =
//            (-0.00541356 * cloud_filtered->points[i].x + 0.0257402 * cloud_filtered->points[i].y + 5.28737) / -0.999654 + 0.3;
//        cloud_filtered->points[i].z = repaired_dis_z;
//      }

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
    sor.setMeanK(20);
    sor.setStddevMulThresh(2);
    sor.filter(*filter);
    *cloud_filtered = *filter;
  }
}
void AdjustCloud::euclideanCluster() {
  pcl::search::KdTree<pcl::PointXYZ>::Ptr tree(new pcl::search::KdTree<pcl::PointXYZ>);
  tree->setInputCloud(cloud_filtered);
  std::vector<pcl::PointIndices> cluster_indices;
  pcl::EuclideanClusterExtraction<pcl::PointXYZ> ec;
  ec.setClusterTolerance(1); //设置近邻搜索的搜索半径为2cm
  ec.setMinClusterSize(100);//设置一个聚类需要的最少点数目为100
  ec.setMaxClusterSize(25000); //设置一个聚类需要的最大点数目为25000
  ec.setSearchMethod(tree);//设置点云的搜索机制
  ec.setInputCloud(cloud_filtered);
  ec.extract(cluster_indices);//从点云中提取聚类，并将点云索引保存在cluster_indices中
  int color_bar[][3] =
      {
          {255, 0, 0},
          {0, 255, 0},
          {0, 0, 255},
          {0, 255, 255},
          {255, 255, 0},
          {255, 255, 255},
          {255, 0, 255}
      };
  int k = 0;
  int j = 0;
  cloud_clustered->points.resize(cloud_filtered->size());

  for (std::vector<pcl::PointIndices>::const_iterator it = cluster_indices.begin(); it != cluster_indices.end(); it++) {
    float min_x = std::numeric_limits<float>::max();
    float min_y = std::numeric_limits<float>::max();
    float max_x = -std::numeric_limits<float>::max();
    float max_y = -std::numeric_limits<float>::max();
    int min_x_index, max_x_index, min_y_index, max_y_index;
    for (std::vector<int>::const_iterator pit = it->indices.begin(); pit != it->indices.end(); pit++) {
      {
        cloud_clustered->points[j].x = cloud_filtered->points[*pit].x;
        cloud_clustered->points[j].y = cloud_filtered->points[*pit].y;
        cloud_clustered->points[j].z = cloud_filtered->points[*pit].z;
        cloud_clustered->points[j].r = color_bar[k][0];
        cloud_clustered->points[j].g = color_bar[k][1];
        cloud_clustered->points[j].b = color_bar[k][2];
        j++;
      }
      {
        if (cloud_filtered->points[*pit].x < min_x) {
          min_x = cloud_filtered->points[*pit].x;
          min_x_index = *pit;
        }
        if (cloud_filtered->points[*pit].y < min_y) {
          min_y = cloud_filtered->points[*pit].y;
          min_y_index = *pit;
        }
        if (cloud_filtered->points[*pit].x > max_x) {
          max_x = cloud_filtered->points[*pit].x;
          max_x_index = *pit;
        }
        if (cloud_filtered->points[*pit].y > max_y) {
          max_y = cloud_filtered->points[*pit].y;
          max_y_index = *pit;
        }
      }
    }
    {
      min_x_point_cluster.push_back(std::make_pair(k, cloud_filtered->points[min_x_index]));
      max_x_point_cluster.push_back(std::make_pair(k, cloud_filtered->points[max_x_index]));
      min_y_point_cluster.push_back(std::make_pair(k, cloud_filtered->points[min_y_index]));
      max_y_point_cluster.push_back(std::make_pair(k, cloud_filtered->points[max_y_index]));
    }
    k++;
    if (k > 6) k = 0;
  }
}
void AdjustCloud::pubROIMarker() {
  if (!min_x_point_cluster.empty() && !max_x_point_cluster.empty()
      && !min_y_point_cluster.empty() && !max_y_point_cluster.empty()) {
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
    float z =
        (-0.00541356 * min_x_point_cluster[0].second.x + 0.0257402 * min_x_point_cluster[0].second.y + 5.28737)
            / -0.999654
            + 0.3;
    geometry_msgs::Point point_ll, point_lu, point_lr, point_ur;
    geometry_msgs::Point point_1, point_2;
    for (int i = 0; i < min_x_point_cluster.size(); i++) {
      point_ll.x = min_x_point_cluster[i].second.x;
      point_ll.y = min_y_point_cluster[i].second.y;
      point_ll.z = z;
      point_lu.x = min_x_point_cluster[i].second.x;
      point_lu.y = max_y_point_cluster[i].second.y;
      point_lu.z = z;
      point_lr.x = max_x_point_cluster[i].second.x;
      point_lr.y = min_y_point_cluster[i].second.y;
      point_lr.z = z;
      point_ur.x = max_x_point_cluster[i].second.x;
      point_ur.y = max_y_point_cluster[i].second.y;
      point_ur.z = z;
      line_list.points.push_back(point_ll);
      line_list.points.push_back(point_lu);
      line_list.points.push_back(point_lu);
      line_list.points.push_back(point_ur);
      line_list.points.push_back(point_ur);
      line_list.points.push_back(point_lr);
      line_list.points.push_back(point_lr);
      line_list.points.push_back(point_ll);

      for (float x_1 = min_x_point_cluster[i].second.x; x_1 <= max_x_point_cluster[i].second.x;) {
        point_1.x = x_1;
        point_1.y = min_y_point_cluster[i].second.y;
        point_1.z = z;

        point_2.x = x_1;
        point_2.y = max_y_point_cluster[i].second.y;
        point_2.z = z;

        x_1 += 0.1;
        line_list.points.push_back(point_1);
        line_list.points.push_back(point_2);
      }
      for (float y_1 = min_y_point_cluster[i].second.y; y_1 <= max_y_point_cluster[i].second.y;) {
        point_1.x = min_x_point_cluster[i].second.x;
        point_1.y = y_1;
        point_1.z = z;

        point_2.x = max_x_point_cluster[i].second.x;
        point_2.y = y_1;
        point_2.z = z;

        y_1 += 0.1;
        line_list.points.push_back(point_1);
        line_list.points.push_back(point_2);
      }
      marker_pub.publish(line_list);
    }
  }
}
void AdjustCloud::gridROI() {
  if (!min_x_point_cluster.empty() && !max_x_point_cluster.empty()
      && !min_y_point_cluster.empty() && !max_y_point_cluster.empty()) {
    float x_start, y_start, x_next_step, y_next_step;
    float average_hight, total_hight;
    int num;
    float z = (-0.00541356 * min_x_point_cluster[0].second.x + 0.0257402 * min_x_point_cluster[0].second.y + 5.28737)
        / -0.999654
        + 0.3;
    pcl::PointCloud<pcl::PointXYZ>::Ptr copy_cloud(new pcl::PointCloud<pcl::PointXYZ>);
    pcl::copyPointCloud(*cloud_filtered, *copy_cloud);
    for (int i = 0; i < min_x_point_cluster.size(); i++) {
//      cout << "x_long " << max_x_point_cluster[i].second.x - min_x_point_cluster[i].second.x << endl;
//      cout << "y_long " << max_y_point_cluster[i].second.y - min_y_point_cluster[i].second.y << endl;
      x_start = min_x_point_cluster[i].second.x;
      average_hight = 0;
      num = 0;
      for (; x_start <= max_x_point_cluster[i].second.x; x_start += 0.1) {
        y_start = min_y_point_cluster[i].second.y;
        x_next_step = x_start + 0.1;
        for (; y_start <= max_y_point_cluster[i].second.y; y_start += 0.1) {
          y_next_step = y_start + 0.1;
          for (std::vector<pcl::PointXYZ, Eigen::aligned_allocator<pcl::PointXYZ>>::iterator
                   it = cloud_filtered->points.begin();
               it != cloud_filtered->points.end(); it++) {
            if (it->x > x_start && it->x < x_next_step) {
              if (it->y > y_start && it->y < y_next_step) {
                total_hight += it->z - z;
                num += 1;
                copy_cloud->points.erase(it);
              }
            }
          }
          if (num != 0) {
            average_hight = total_hight / num;
            hight_vec.push_back(average_hight);
            num = 0;
            total_hight = 0.0;
            average_hight = 0.0;
          }
        }
      }
    }
  }
  float sum = 0;
  cout << "sum " << std::accumulate(hight_vec.begin(), hight_vec.end(), sum) << endl;
  cout << "hight size " << hight_vec.size() << endl;
}

int
main(int argc, char **argv) {
  ros::init(argc, argv, "AdjustCloud");
  pcl::PointCloud<pcl::PointXYZ>::Ptr source_ptr(new pcl::PointCloud<pcl::PointXYZ>());
  AdjustCloud trans_pointcloud(source_ptr);
  ros::Rate sleep_rate(5);
  trans_pointcloud.loadSourceCloud();

  Eigen::Vector3d euler_angle(-82 * PI / 180, -3.5 * PI / 180, 0);
  Eigen::Vector3d trans_vector(108, 9.5, -3);
  Eigen::AngleAxisd rotation_vector;
  rotation_vector = Eigen::AngleAxisd(euler_angle[0], Eigen::Vector3d::UnitZ()) *
      Eigen::AngleAxisd(euler_angle[1], Eigen::Vector3d::UnitY()) *
      Eigen::AngleAxisd(euler_angle[2], Eigen::Vector3d::UnitX());
  Eigen::Isometry3d T = Eigen::Isometry3d::Identity();
  T.rotate(rotation_vector);
  T.pretranslate(trans_vector);
  Eigen::Matrix4d rotation_matrix = T.matrix();
  vector<std::pair<int, float>> dis_vec;
  trans_pointcloud.filterPointByPlane(dis_vec);
  //trans_pointcloud.repairModel(dis_vec);
  trans_pointcloud.removeOutlier();
  trans_pointcloud.euclideanCluster();
  trans_pointcloud.gridROI();
  while (ros::ok()) {
    //trans_pointcloud.broadcasterTF();
//    Eigen::Matrix4d transform = trans_pointcloud.getTransform();
//    trans_pointcloud.transformCloud(transform);
    trans_pointcloud.transformCloud(rotation_matrix);
//    trans_pointcloud.cloudAlign();
    trans_pointcloud.pubROIMarker();
    trans_pointcloud.pubTransformedCloud();
    sleep_rate.sleep();
    ros::spinOnce();
  }
}
