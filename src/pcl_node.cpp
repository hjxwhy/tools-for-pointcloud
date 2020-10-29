#include <iostream>
#include <ros/ros.h>
#include <pcl/point_cloud.h>
#include <pcl_conversions/pcl_conversions.h>
#include <sensor_msgs/PointCloud2.h>
#include <visualization_msgs/Marker.h>
#include <interactive_markers/interactive_marker_server.h>
#include <boost/circular_buffer.hpp>
#include <pcl/common/transforms.h>
#include <pcl/visualization/pcl_visualizer.h>
#include <cv_bridge/cv_bridge.h>
#include <opencv2/highgui/highgui.hpp>
#include <pcl/console/parse.h>
#include <image_transport/image_transport.h>
#include <dynamic_reconfigure/server.h>
#include <std_msgs/Bool.h>
#include <geometry_msgs/PointStamped.h>
#include <stdio.h>

#include "detect_drive/TutorialsConfig.h"

using namespace std;
using namespace boost;
using namespace cv;

static detect_drive::TutorialsConfig g_my_config;
static pcl::PointCloud<pcl::PointXYZI> g_source_PointCloud;
static pcl::PointCloud<pcl::PointXYZI> g_transed_PointCloud;
static pcl::PointCloud<pcl::PointXYZI> g_sumPointCloud;
static geometry_msgs::Pose g_pre_pose;
static bool g_pcl_source_flag = false;
static bool g_pcl_transed_flag = false;
static bool g_dy_flag = false;

void init_interactive_marker(visualization_msgs::InteractiveMarker &int_marker,
                             string frame_id,
                             string name,
                             string des)  //交互式marker
{
  int_marker.header.frame_id = frame_id;
  int_marker.header.stamp = ros::Time::now();
  int_marker.name = name;
  int_marker.description = des;
}

void init_marker(visualization_msgs::Marker &marker,
                 double x,
                 double y,
                 double z,
                 float a,
                 float r,
                 float g,
                 float b)  //普通marker
{
  marker.type = visualization_msgs::Marker::CUBE;  //正方体
  marker.scale.x = x;
  marker.scale.y = y;
  marker.scale.z = z;
  marker.color.a = a;
  marker.color.r = r;
  marker.color.g = g;
  marker.color.b = b;
}

void processFeedback(const visualization_msgs::InteractiveMarkerFeedbackConstPtr &feedback) {
  if (feedback != NULL) {
    ROS_INFO_STREAM(feedback->marker_name << "  is  now  at  "
                                          << feedback->pose.position.x << ",  " << feedback->pose.position.y
                                          << ",  " << feedback->pose.position.z);
    g_pre_pose = feedback->pose;
  }
}

void call_back_source(const sensor_msgs::PointCloud2ConstPtr &input)    //订阅回调函数
{
  if (input != NULL) {
    pcl::fromROSMsg(*input, g_source_PointCloud);
    g_pcl_source_flag = true;
  }
}

void call_back_transed(const sensor_msgs::PointCloud2ConstPtr &input) {
  if (input != NULL) {
    pcl::fromROSMsg(*input, g_transed_PointCloud);
    g_pcl_transed_flag = true;
  }
}

void pub_transform_pcl(string frame_id, ros::Publisher &pub) {
  pcl::PointCloud<pcl::PointXYZI> cloud_out, transformed_cloud;
  Eigen::Affine3f transform = Eigen::Affine3f::Identity();
  transform.translation()
      << float(g_my_config.distence_x), float(g_my_config.distence_y), float(g_my_config.distence_z);

  transform.rotate(Eigen::AngleAxisf(float(g_my_config.theta_x), Eigen::Vector3f::UnitX()));
  transform.rotate(Eigen::AngleAxisf(float(g_my_config.theta_y), Eigen::Vector3f::UnitY()));
  transform.rotate(Eigen::AngleAxisf(float(g_my_config.theta_z), Eigen::Vector3f::UnitZ()));
  pcl::transformPointCloud(g_source_PointCloud, transformed_cloud, transform);

  sensor_msgs::PointCloud2 tran_cloud;
  pcl::toROSMsg(transformed_cloud, tran_cloud);
  tran_cloud.header.frame_id = frame_id;
  tran_cloud.header.stamp = ros::Time::now();
  pub.publish(tran_cloud);
}

/**
* @brief pcl_func
* @param frame_id
* @param pub
* @return
* 激光雷达转换后的点云通过与ROI标记marker的长宽高进行比较来提取出目标点云并输出发布ROI点云数据
**/

int pcl_func(string frame_id, ros::Publisher &pub) {
  static circular_buffer<pcl::PointCloud<pcl::PointXYZI> > cicPointCloud(50);
  pcl::PointCloud<pcl::PointXYZI> cloud_out, sum;

  for (unsigned int i = 0; i < g_transed_PointCloud.points.size(); i++) {
    if (g_transed_PointCloud.points[i].x <= float(g_pre_pose.position.x + g_my_config.scale_x / 2) &&
        g_transed_PointCloud.points[i].x >= float(g_pre_pose.position.x - g_my_config.scale_x / 2) &&
        g_transed_PointCloud.points[i].y <= float(g_pre_pose.position.y + g_my_config.scale_y / 2) &&
        g_transed_PointCloud.points[i].y >= float(g_pre_pose.position.y - g_my_config.scale_y / 2) &&
        g_transed_PointCloud.points[i].z <= float(g_pre_pose.position.z + g_my_config.scale_z / 2) &&
        g_transed_PointCloud.points[i].z >= float(g_pre_pose.position.z - g_my_config.scale_z / 2)) {
      cloud_out.push_back(g_transed_PointCloud.points[i]);
    }
  }
  cicPointCloud.push_back(cloud_out);

  for (unsigned long i = 0; i < cicPointCloud.size(); i++) {
    sum += cicPointCloud.at(i);
  }
  g_sumPointCloud = sum;

  sensor_msgs::PointCloud2 ros_cloud;
  pcl::toROSMsg(sum, ros_cloud);
  ros_cloud.header.frame_id = frame_id;
  ros_cloud.header.stamp = ros::Time::now();
  pub.publish(ros_cloud);
  return int(sum.size());
}

/**
* @brief cut_image
* @param resize
* @param pub
* @return
*　将3d点云绘制到2d图像，并显示释放时拨叉的伪RGB图
**/
int cut_image(int resize, ros::Publisher &pub) {
  int cols = int(g_my_config.scale_y * resize);
  int rows = int(g_my_config.scale_z * resize);
  Mat roi_image = Mat(rows, cols, CV_8UC1, Scalar(0));
  vector<float> array;
  for (unsigned int i = 0; i < g_sumPointCloud.points.size(); i++) {
    if (g_sumPointCloud.points[i]._PointXYZI::intensity <= g_my_config.intensity_min) {
      array.push_back(255 - g_sumPointCloud.points[i]._PointXYZI::intensity);
    }
  }
  normalize(array, array, 0, 255, cv::NORM_MINMAX); //归一化
  int k = 0;
  for (unsigned int i = 0; i < g_sumPointCloud.points.size(); i++) {
    float col = (float(g_pre_pose.position.y) - g_sumPointCloud.points[i].y) * resize;
    float row = (float(g_pre_pose.position.z) - g_sumPointCloud.points[i].z) * resize;
    if (col >= 0 && row >= 0) {
      col = cols / 2 + col;
      row = rows / 2 - row;
    } else if (col < 0 && row >= 0) {
      col = cols / 2 - abs(col);
      row = rows / 2 - row;
    } else if (col < 0 && row < 0) {
      col = cols / 2 - abs(col);
      row = rows / 2 + abs(row);
    } else if (col >= 0 && row < 0) {
      col = cols / 2 + col;
      row = rows / 2 + abs(row);
    }
    if (col > 0 && col < cols && row > 0 && row < rows) {
      row = rows - row;
      if (g_sumPointCloud.points[i]._PointXYZI::intensity <= g_my_config.intensity_min) {
        roi_image.at<uchar>(int(row), int(col)) = array.at(k);
        ++k;
      }
    }
  }

  applyColorMap(roi_image, roi_image, COLORMAP_JET);  //伪彩色

  if (!roi_image.empty()) {
    imshow("roi_image", roi_image);
    waitKey(10);
    sensor_msgs::ImagePtr msg = cv_bridge::CvImage(std_msgs::Header(), "bgr8", roi_image).toImageMsg();
    pub.publish(msg);
  }
  return int(array.size());
}

void dynamic_callback(detect_drive::TutorialsConfig &config) {
  g_my_config = config;
  ROS_INFO_STREAM(g_my_config.scale_x << " "
                                      << g_my_config.scale_y << " "
                                      << g_my_config.scale_z << " "
                                      << g_my_config.color_a << " "
                                      << g_my_config.theta_x << " "
                                      << g_my_config.theta_y << " "
                                      << g_my_config.theta_z << " "
                                      << g_my_config.distence_x << " "
                                      << g_my_config.distence_y << " "
                                      << g_my_config.distence_z
  );
  g_dy_flag = true;
}

void change_marker(visualization_msgs::InteractiveMarker &int_marker) {
  int_marker.controls.at(0).markers.at(0).scale.x = g_my_config.scale_x;
  int_marker.controls.at(0).markers.at(0).scale.y = g_my_config.scale_y;
  int_marker.controls.at(0).markers.at(0).scale.z = g_my_config.scale_z;
  int_marker.controls.at(0).markers.at(0).color.a = float(g_my_config.color_a);
  int_marker.pose = g_pre_pose;
}

static bool click_flag = false;

void callback_clicked(const geometry_msgs::PointStampedConstPtr &clicked) {
  if (clicked != NULL) {
    g_pre_pose.position.x = clicked->point.x + (g_my_config.scale_x / 2.0);
    g_pre_pose.position.y = clicked->point.y - 0.035;
    g_pre_pose.position.z = clicked->point.z - 0.035;
    cout << "x: " << g_pre_pose.position.x << "  y: " << g_pre_pose.position.y << "   z: " << g_pre_pose.position.z
         << endl;
    click_flag = true;
  }
}

int main(int argc, char **argv) {
  ros::init(argc, argv, "pcl_node");
  ros::NodeHandle nh;

  ros::Subscriber pcl_sub = nh.subscribe(argv[1], 1, call_back_source);

  ros::Publisher trans_pub = nh.advertise<sensor_msgs::PointCloud2>("/transfromed_bocha", 1);

  ros::Subscriber trans_sub = nh.subscribe("/transfromed_bocha", 1, call_back_transed);

  ros::Publisher pcl_pub = nh.advertise<sensor_msgs::PointCloud2>("/roi_bocha", 1);

  ros::Publisher image_pub = nh.advertise<sensor_msgs::Image>("/roi_image_bocha", 1);

  ros::Publisher bocha_state_pub = nh.advertise<std_msgs::Bool>("/bocha_state", 100);

  ros::Subscriber click_sub = nh.subscribe("/clicked_point", 1, callback_clicked);

  interactive_markers::InteractiveMarkerServer server("sbocha_marker");

  dynamic_reconfigure::Server<detect_drive::TutorialsConfig> dy_server;
  dynamic_reconfigure::Server<detect_drive::TutorialsConfig>::CallbackType callback;

  callback = boost::bind(&dynamic_callback, _1);
  dy_server.setCallback(callback);

  visualization_msgs::InteractiveMarker int_marker;
  init_interactive_marker(int_marker, "livox_frame", "bocha_marker", "bocha Control");

  visualization_msgs::Marker box_marker;
  init_marker(box_marker, 0.1, 1, 1, float(0.1), 1, 1, 1);

  visualization_msgs::InteractiveMarkerControl box_control; // create a non-interactive control which contains the box
  box_control.always_visible = true;
  box_control.markers.push_back(box_marker);

  visualization_msgs::InteractiveMarkerControl rotate_control_x; //控制组件,沿局部x轴平移
  rotate_control_x.name = "move_x";
  rotate_control_x.interaction_mode =
      visualization_msgs::InteractiveMarkerControl::MOVE_AXIS;

  visualization_msgs::InteractiveMarkerControl rotate_control_yz; //控制组件,沿yz平面中平移
  rotate_control_yz.name = "move_yz";
  rotate_control_yz.interaction_mode =
      visualization_msgs::InteractiveMarkerControl::MOVE_PLANE;

  int_marker.controls.push_back(box_control);   // add the control to the interactive marker
  int_marker.controls.push_back(rotate_control_x);
  int_marker.controls.push_back(rotate_control_yz);

  g_pre_pose.position.x = g_my_config.p_x;
  g_pre_pose.position.y = g_my_config.p_y;
  g_pre_pose.position.z = g_my_config.p_z;

  server.insert(int_marker, &processFeedback);
  server.setPose("bocha_marker", g_pre_pose);
  server.applyChanges();
  ros::Rate loop_rate(10);

  while (ros::ok()) {
    if (g_pcl_source_flag) {
      pub_transform_pcl("livox_frame", trans_pub);
      g_pcl_source_flag = false;
    }
    if (g_pcl_transed_flag) {
      pcl_func("livox_frame", pcl_pub);
      int bocha_size = cut_image(g_my_config.image_size, image_pub);
      std_msgs::Bool bocha_msg;
      if (bocha_size >= g_my_config.pcl_size) {
        cout << "bocha1 size: " << bocha_size << endl;
        bocha_msg.data = true;
        bocha_state_pub.publish(bocha_msg);
        cout << "拨叉1出现" << endl;
      } else {
        // cout << "bocha1 size: " << bocha_size << endl;
        // cout << "拨叉1消失" << endl;
        bocha_msg.data = false;
        bocha_state_pub.publish(bocha_msg);
      }
      g_pcl_transed_flag = false;
    }
    if (g_dy_flag) {
      server.erase("bocha_marker");
      server.applyChanges();
      change_marker(int_marker);
      server.insert(int_marker, &processFeedback);
      server.setPose("bocha_marker", g_pre_pose);
      server.applyChanges();
      g_dy_flag = false;
    }
    if (click_flag) {
      click_flag = false;
      server.erase("bocha_marker");
      server.applyChanges();
      change_marker(int_marker);
      server.insert(int_marker, &processFeedback);
      server.setPose("bocha_marker", g_pre_pose);
      server.applyChanges();
    }

    loop_rate.sleep();
    ros::spinOnce();
  }
  return 0;
}




