#include <ros/ros.h>
#include <boost/bind.hpp>
#include <Eigen/Core>
#include <opencv2/core/eigen.hpp>
#include <opencv2/opencv.hpp>
#include <sensor_msgs/Image.h>
#include <sensor_msgs/CompressedImage.h>
#include <sensor_msgs/PointCloud2.h>
#include <std_msgs/Header.h>
#include <message_filters/subscriber.h>
#include <message_filters/synchronizer.h>
#include <message_filters/sync_policies/approximate_time.h>
#include <message_filters/sync_policies/exact_time.h>
#include <image_transport/image_transport.h>
#include <sensor_msgs/image_encodings.h>
#include <cv_bridge/cv_bridge.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl_conversions/pcl_conversions.h>
#include <pcl/common/transforms.h>	
#include <pcl/visualization/pcl_visualizer.h>
#include <pcl/io/pcd_io.h>
#include <math.h>
#include "colored_pointcloud/colored_pointcloud.h"
#include "ndt_cpu/NormalDistributionsTransform.h"
#include <sys/stat.h>
#include <sys/types.h> 
#include <cstdio>
#include <ctime>

#define YELLOW "\033[33m" /* Yellow */
#define GREEN "\033[32m"  /* Green */
#define REND "\033[0m" << std::endl

#define WARN (std::cout << YELLOW)
#define INFO (std::cout << GREEN)

ros::Publisher fused_image_pub, colored_cloud_showpub, cloudmap;
Eigen::Matrix4f current_pose = Eigen::Matrix4f::Identity();
Eigen::Matrix4f last_pose = Eigen::Matrix4f::Identity();
pcl::PointCloud<pcl::PointXYZ>::Ptr point_map(new pcl::PointCloud<pcl::PointXYZ>);
pcl::PointCloud<pcl::PointXYZRGB>::Ptr colored_cloud_toshow (new pcl::PointCloud<pcl::PointXYZRGB>);
bool initial = true;

std::ofstream pose_pic;
int img_count = 0;
cv::Mat ProjectionMat;

cv::Mat cam2Tocam0R, cam2Tocam0T;


// colored_cloud_pub;

//we suppose that lidar and camera have two different coordinate system and the tranjectory of camera
//in the space is actually the same as the lidar


class RsCamFusion
{
  private:
    cv::Mat intrinsic;
    cv::Mat extrinsic;
    cv::Mat distcoeff;
    cv::Size imageSize;
    Eigen::Matrix4d transform, inv_transform;
    cv::Mat rVec = cv::Mat::zeros(3, 1, CV_64FC1); // Rotation vector
    cv::Mat rMat = cv::Mat::eye(3, 3, CV_64FC1);
    cv::Mat tVec = cv::Mat::zeros(3, 1, CV_64FC1); // Translation vector

    Eigen::Matrix3d rel_R2;
    Eigen::Vector3d rel_T2;



    Eigen::Matrix4d rel_RT = Eigen::Matrix4d::Identity();
    Eigen::Matrix4d PrjMat = Eigen::Matrix4d::Identity();
 

    bool show_colored_cloud, save_data;
    std::string image_save_dir, cloud_save_dir, colored_cloud_save_dir;

    int color[21][3] = 
    {
        {255, 0, 0}, {255, 69, 0}, {255, 99, 71}, 
        {255, 140, 0}, {255, 165, 0}, {238, 173, 14},
        {255, 193, 37}, {255, 255, 0}, {255, 236, 139},
        {202, 255, 112}, {0, 255, 0}, {84, 255, 159},
        {127, 255, 212}, {0, 229, 238}, {152, 245, 255},
        {178, 223, 238}, {126, 192, 238}, {28, 134, 238},
        {0, 0, 255}, {72, 118, 255}, {122, 103, 238} 
    };
    float color_distance;   //step length to color the lidar points according to plane distance(z)
    int frame_count = 0;

  public:
    RsCamFusion(cv::Mat cam_intrinsic, cv::Mat lidar2cam_extrinsic, cv::Mat cam_distcoeff, cv::Size img_size, float color_dis, bool show_cloud, bool save)
    {
      intrinsic = cam_intrinsic;
      extrinsic = lidar2cam_extrinsic;
      distcoeff = cam_distcoeff;

      for(int i = 0; i < 4; i++)
      {
          for(int j = 0; j < 4; j++)
          {
              transform(i,j) = extrinsic.at<double>(i,j);
          }
      }
      std::cout<<transform<<std::endl;

      for(int i = 0; i < 3; i++)
      {
          for(int j = 0; j < 4; j++)
          {
              PrjMat(i,j) = ProjectionMat.at<double>(i,j);
              
          }
      }
      for(int i = 0; i < 3; i++)
      {
          for(int j = 0; j < 3; j++)
          {
              rel_R2(i,j) = cam2Tocam0R.at<double>(i,j);
          }
      }

        
      std::cout<<PrjMat<<std::endl;
      std::cout<<rel_R2<<std::endl;

      rel_T2(0,0) = cam2Tocam0T.at<double>(0);
      rel_T2(1,0) = cam2Tocam0T.at<double>(1);
      rel_T2(2,0) = cam2Tocam0T.at<double>(2);

      std::cout<<rel_T2<<std::endl;

      rel_RT.block<3,3>(0,0) = rel_R2;
      rel_RT.block<3,1>(0,3) = rel_T2;
      transform = rel_RT*transform;
      inv_transform = transform.inverse();
      imageSize = img_size;
      color_distance = color_dis;
      show_colored_cloud = show_cloud;
      save_data = save;
      if(save_data)
      {
        time_t rawtime;
        struct tm *ptminfo;
        time(&rawtime);
        ptminfo = localtime(&rawtime);
        std::string currentdate = "/data/" + std::to_string(ptminfo->tm_year + 1900) + std::to_string(ptminfo->tm_mon + 1) 
                                          + std::to_string(ptminfo->tm_mday) + std::to_string(ptminfo->tm_hour) 
                                          + std::to_string(ptminfo->tm_min) + std::to_string(ptminfo->tm_sec);
        mkdir(currentdate.c_str(),S_IRUSR | S_IWUSR | S_IXUSR | S_IRWXG | S_IRWXO);
        image_save_dir = "/home/lie/msf_ws/front_camera";
        mkdir(image_save_dir.c_str(),S_IRUSR | S_IWUSR | S_IXUSR | S_IRWXG | S_IRWXO);
        cloud_save_dir = "/home/lie/msf_ws/rslidar_points";
        mkdir(cloud_save_dir.c_str(),S_IRUSR | S_IWUSR | S_IXUSR | S_IRWXG | S_IRWXO);
        colored_cloud_save_dir = "/home/lie/msf_ws/colored_cloud";
        mkdir(colored_cloud_save_dir.c_str(),S_IRUSR | S_IWUSR | S_IXUSR | S_IRWXG | S_IRWXO);
      }
    }

    void callback(const sensor_msgs::ImageConstPtr input_image_msg,
                const sensor_msgs::PointCloud2ConstPtr input_cloud_msg)
    {
      cv::Mat input_image;
      cv::Mat undistorted_image;
      cv_bridge::CvImagePtr cv_ptr; 

      std_msgs::Header image_header = input_image_msg->header;
      std_msgs::Header cloud_header = input_cloud_msg->header;
      // INFO << image_header << REND;

    // sensor_msgs to cv image   
      try
      {
        cv_ptr = cv_bridge::toCvCopy(input_image_msg, sensor_msgs::image_encodings::BGR8);
      }
      catch(cv_bridge::Exception e)
      {
        ROS_ERROR_STREAM("Cv_bridge Exception:"<<e.what());
        return;
      }
      input_image = cv_ptr->image;
      
      //sensor_msgs to pointxyzi
      pcl::PointCloud<pcl::PointXYZ>::Ptr tmp_cloud(new pcl::PointCloud<pcl::PointXYZ>);
      pcl::PointCloud<pcl::PointXYZI>::Ptr input_cloud_ptr(new pcl::PointCloud<pcl::PointXYZI>);
      pcl::PointCloud<pcl::PointXYZI>::Ptr transformed_cloud (new pcl::PointCloud<pcl::PointXYZI>);
      pcl::fromROSMsg(*input_cloud_msg, *tmp_cloud);
      if (tmp_cloud->size() == 0)
      {
        WARN << "input cloud is empty, please check it out!" << REND;
      }

      //use tmp_cloud to scan matching, get the current pose of laserlidar and trnasform it to camera link later
      if(initial)
      {
        *point_map += *tmp_cloud;
        initial = false;
      }
      else
      {
        cpu::NormalDistributionsTransform<pcl::PointXYZ, pcl::PointXYZ> ndt_optimizer;
        ndt_optimizer.setResolution(2);
        ndt_optimizer.setMaximumIterations(30);
        ndt_optimizer.setStepSize(0.1);
        ndt_optimizer.setTransformationEpsilon(0.01);

        pcl::VoxelGrid<pcl::PointXYZ> downSampler;
        pcl::VoxelGrid<pcl::PointXYZ> lastdownSampler;


        pcl::PointCloud<pcl::PointXYZ>::Ptr filtered_source(new pcl::PointCloud<pcl::PointXYZ>);
        downSampler.setLeafSize(0.1,0.1,0.1);
        //downsample
        downSampler.setInputCloud(tmp_cloud);
        downSampler.filter(*filtered_source);

        pcl::PointCloud<pcl::PointXYZ>::Ptr filtered_target(new pcl::PointCloud<pcl::PointXYZ>);
        lastdownSampler.setLeafSize(0.1,0.1,0.1);

        //downsample
        lastdownSampler.setInputCloud(point_map);
        lastdownSampler.filter(*filtered_target);

        ndt_optimizer.setInputTarget(filtered_target);
        ndt_optimizer.setInputSource(filtered_source);
        ndt_optimizer.align(last_pose);

        current_pose = ndt_optimizer.getFinalTransformation();
        double trans_probability = ndt_optimizer.getTransformationProbability();

        std::cout<<img_count<<std::endl;
        std::cout<<"----------trans_probablity is"<<trans_probability<<std::endl;

        if(trans_probability<2.0)
        {
            current_pose = last_pose;
        }

        pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_to_map(new pcl::PointCloud<pcl::PointXYZ>);

        pcl::transformPointCloud (*tmp_cloud, *cloud_to_map, current_pose); 

        *point_map += *cloud_to_map;

        img_count++;
      }


      publishCloudMap(cloudmap,cloud_header,point_map);

      for(int i=0;i<tmp_cloud->points.size();i++)
      {
        pcl::PointXYZI point_int;
        point_int.x = tmp_cloud->points[i].x;
        point_int.y = tmp_cloud->points[i].y;
        point_int.z = tmp_cloud->points[i].z;
        point_int.intensity = 0.0;
        input_cloud_ptr->points.push_back(point_int);
      }

      //transform lidar points from lidar local coordinate to camera local coordinate
      pcl::transformPointCloud (*input_cloud_ptr, *transformed_cloud, transform);        //lidar coordinate(forward x+, left y+, up z+) 
                                                                                         //camera coordiante(right x+, down y+, forward z+) (3D-3D)  
                                                                                         //using the extrinsic matrix between this two coordinate system

      std::vector<cv::Point3d> lidar_points;
      std::vector<cv::Scalar> dis_color;
      std::vector<float> intensity;
      std::vector<cv::Point2d> imagePoints;
      pcl::PointCloud<pcl::PointXYZI>::Ptr transformed_cloud_imgplane (new pcl::PointCloud<pcl::PointXYZI>);
      
      //reserve the points in front of the camera(z>0)
      for(int i=0;i<=transformed_cloud->points.size();i++)
      {
          if(transformed_cloud->points[i].z>0)
          {
            lidar_points.push_back(cv::Point3d(transformed_cloud->points[i].x, transformed_cloud->points[i].y, transformed_cloud->points[i].z));
            int color_order = int(transformed_cloud->points[i].z / color_distance);
            if(color_order > 20)
            {
              color_order = 20;
            }
            dis_color.push_back(cv::Scalar(color[color_order][2], color[color_order][1], color[color_order][0]));
            //intensity.push_back(transformed_cloud->points[i].intensity);
            intensity.push_back(transformed_cloud->points[i].intensity);
            transformed_cloud_imgplane->points.push_back(transformed_cloud->points[i]);
          }
      }


      //project world points from the camera coordinate to the image coordinate(right x+, down y+)  
      pcl::transformPointCloud (*transformed_cloud_imgplane, *transformed_cloud_imgplane, PrjMat);  

      for(int i=0;i<transformed_cloud_imgplane->points.size();i++)
      {
        cv::Point2d imgpoint;
        imgpoint.x = transformed_cloud_imgplane->points[i].x/transformed_cloud_imgplane->points[i].z;
        imgpoint.y = transformed_cloud_imgplane->points[i].y/transformed_cloud_imgplane->points[i].z;
        imagePoints.push_back(imgpoint);
      }    
      
      pcl::PointCloud<PointXYZRGBI>::Ptr colored_cloud (new pcl::PointCloud<PointXYZRGBI>);
      pcl::PointCloud<PointXYZRGBI>::Ptr colored_cloud_transback (new pcl::PointCloud<PointXYZRGBI>);
      cv::Mat image_to_show = input_image.clone();

      for(int i=0;i<imagePoints.size();i++)
      {
        if(imagePoints[i].x>=0 && imagePoints[i].x<imageSize.width && imagePoints[i].y>=0 && imagePoints[i].y<imageSize.height)
        {
          cv::circle(image_to_show, imagePoints[i], 1, dis_color[i], 2, 8, 0);
          PointXYZRGBI point;                                                             //reserve the lidar points in the range of image 
          point.x = lidar_points[i].x;                                                        //use 3D lidar points and RGB value of the corresponding pixels  
          point.y = lidar_points[i].y;                                                        //to create colored point clouds
          point.z = lidar_points[i].z;
          point.r = input_image.at<cv::Vec3b>(imagePoints[i].y, imagePoints[i].x)[2];
          point.g = input_image.at<cv::Vec3b>(imagePoints[i].y, imagePoints[i].x)[1];
          point.b = input_image.at<cv::Vec3b>(imagePoints[i].y, imagePoints[i].x)[0];
          point.i = intensity[i];
          colored_cloud->points.push_back(point);  
        }
      }
      publishImage(fused_image_pub, image_header, image_to_show);
      //transform colored points from camera coordinate to lidar coordinate
      pcl::transformPointCloud (*colored_cloud, *colored_cloud_transback, inv_transform);   
      pcl::transformPointCloud (*colored_cloud_transback, *colored_cloud_transback, current_pose);    

      if(show_colored_cloud)
      {  
        pcl::PointCloud<pcl::PointXYZRGB>::Ptr tmp_colored_cloud_toshow (new pcl::PointCloud<pcl::PointXYZRGB>);
        for(int i=0;i<colored_cloud_transback->points.size();i++)
        {
            pcl::PointXYZRGB point;                                                             
            point.x = colored_cloud_transback->points[i].x;                                                        
            point.y = colored_cloud_transback->points[i].y;                                                        
            point.z = colored_cloud_transback->points[i].z;
            point.r = colored_cloud_transback->points[i].r;
            point.g = colored_cloud_transback->points[i].g;
            point.b = colored_cloud_transback->points[i].b;
            tmp_colored_cloud_toshow->points.push_back (point);  
          }
        *colored_cloud_toshow += *tmp_colored_cloud_toshow;
        if(!ros::ok())
        {
            pcl::io::savePCDFile("cloudcolor.pcd",*colored_cloud_toshow);
        }
        publishCloudtoShow(colored_cloud_showpub, cloud_header, colored_cloud_toshow);
      } 

      if(save_data)
      {
        saveData(image_header, input_image, cloud_header, input_cloud_ptr, colored_cloud_transback);
      }
      frame_count = frame_count + 1;
      last_pose = current_pose;
    }

    void publishImage(const ros::Publisher& image_pub, const std_msgs::Header& header, const cv::Mat image)
    {
      cv_bridge::CvImage output_image;
      output_image.header.frame_id = header.frame_id;
      output_image.encoding = sensor_msgs::image_encodings::TYPE_8UC3;
      output_image.image = image;
      image_pub.publish(output_image);
    } 
    
    void saveData(const std_msgs::Header& image_header, const cv::Mat image, const std_msgs::Header& cloud_header,
                      const pcl::PointCloud<pcl::PointXYZI>::Ptr& cloud, pcl::PointCloud<PointXYZRGBI>::Ptr colored_cloud)
    {
      //timestamp: image_header.stamp.sec . image_header.stamp.nsec 
      std::string img_name = std::to_string(frame_count) + "_" + std::to_string(image_header.stamp.sec) + "_" + std::to_string(image_header.stamp.nsec) + ".jpg";
      std::string cloud_name = std::to_string(frame_count) + "_" + std::to_string(cloud_header.stamp.sec) + "_" + std::to_string(cloud_header.stamp.nsec) + ".pcd";
      std::string colored_cloud_name = "c_" + std::to_string(frame_count) + "_" + std::to_string(cloud_header.stamp.sec) + "_" + std::to_string(cloud_header.stamp.nsec) + ".pcd";
      cv::imwrite(image_save_dir + "/" + img_name, image);
      cloud->width = cloud->size();
      cloud->height = 1;
      cloud->is_dense = false;
      cloud->points.resize(cloud->width * cloud->height);
      pcl::io::savePCDFileASCII(cloud_save_dir + "/" + cloud_name, *cloud);

      pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud_no_int(new pcl::PointCloud<pcl::PointXYZRGB>);
      cloud_no_int->points.clear();
      for(int i = 0; i<colored_cloud->size();i++)
      {
        pcl::PointXYZRGB p_color;
        p_color.x = colored_cloud->points[i].x;
        p_color.y = colored_cloud->points[i].y;
        p_color.z = colored_cloud->points[i].z;
        p_color.r = colored_cloud->points[i].r;
        p_color.g = colored_cloud->points[i].g;
        p_color.b = colored_cloud->points[i].b;
        cloud_no_int->points.push_back(p_color);
      }
      cloud_no_int->width = cloud->size();
      cloud_no_int->height = 1;
      cloud_no_int->is_dense = false;
      cloud_no_int->points.resize(cloud->width * cloud->height);
      pcl::io::savePCDFileASCII(colored_cloud_save_dir + "/" + colored_cloud_name, *cloud_no_int);
    }

    void publishCloudtoShow(const ros::Publisher& cloudtoshow_pub, const std_msgs::Header& header,
                      const pcl::PointCloud<pcl::PointXYZRGB>::ConstPtr& cloud)
    {
      // pcl::VoxelGrid<pcl::PointXYZRGB> lastdownSampler;

      // pcl::PointCloud<pcl::PointXYZRGB>::Ptr filtered_output(new pcl::PointCloud<pcl::PointXYZRGB>);
      // lastdownSampler.setLeafSize(0.1,0.1,0.1);
      // lastdownSampler.setInputCloud(cloud);
      // lastdownSampler.filter(*filtered_output);

      sensor_msgs::PointCloud2 output_msg;
      pcl::toROSMsg(*cloud, output_msg);
      output_msg.header = header;
      cloudtoshow_pub.publish(output_msg);
    }

    void publishCloudMap(const ros::Publisher& cloudmap, const std_msgs::Header& header,
                      const pcl::PointCloud<pcl::PointXYZ>::ConstPtr& cloud)
    {
      sensor_msgs::PointCloud2 output_msg;
      pcl::toROSMsg(*cloud, output_msg);
      output_msg.header = header;
      cloudmap.publish(output_msg);
    }
};

int main(int argc, char** argv)
{
  ros::init(argc, argv, "colored_pointcloud_node");
  ros::NodeHandle nh;
  ros::NodeHandle priv_nh("~");

  std::string config_path, file_name;
  std::string camera_topic, lidar_topic;
  float color_dis;
  bool show_cloud, save_data;
  if (priv_nh.hasParam("calib_file_path") && priv_nh.hasParam("file_name"))
  {
    priv_nh.getParam("camera_topic", camera_topic);
    priv_nh.getParam("lidar_topic", lidar_topic);
    priv_nh.getParam("calib_file_path", config_path);
    priv_nh.getParam("file_name", file_name);
    priv_nh.getParam("color_distance", color_dis);
    priv_nh.getParam("show_colored_cloud", show_cloud);
    priv_nh.getParam("save_data", save_data);
  }
  else
  {
    WARN << "Config file is empty!" << REND;
    return 0;
  }
  
  INFO << "config path: " << config_path << REND;
  INFO << "config file: " << file_name << REND;

  std::string config_file_name = config_path + "/" + file_name;
  cv::FileStorage fs_reader(config_file_name, cv::FileStorage::READ);
  
  cv::Mat cam_intrinsic, lidar2cam_extrinsic, cam_distcoeff;
  cv::Size img_size;
  fs_reader["CameraMat"] >> cam_intrinsic;
  fs_reader["CameraExtrinsicMat"] >> lidar2cam_extrinsic;
  fs_reader["DistCoeff"] >> cam_distcoeff;
  fs_reader["ImageSize"] >> img_size;
  fs_reader["ProjectionMat"] >> ProjectionMat;
  fs_reader["R"] >> cam2Tocam0R;
  fs_reader["T"] >> cam2Tocam0T;
  fs_reader.release();

  if (lidar_topic.empty() || camera_topic.empty())
  {
    WARN << "sensor topic is empty!" << REND;
    return 0;
  }

  INFO << "lidar topic: " << lidar_topic << REND;
  INFO << "camera topic: " << camera_topic << REND;
  INFO << "camera intrinsic matrix: " << cam_intrinsic << REND;
  INFO << "lidar2cam extrinsic matrix: " << lidar2cam_extrinsic << REND;
  
  RsCamFusion fusion(cam_intrinsic, lidar2cam_extrinsic, cam_distcoeff, img_size, color_dis, show_cloud, save_data); 
  message_filters::Subscriber<sensor_msgs::Image> camera_sub(nh, camera_topic, 150);
  message_filters::Subscriber<sensor_msgs::PointCloud2> lidar_sub(nh, lidar_topic, 150);
  
  typedef message_filters::sync_policies::ApproximateTime<sensor_msgs::Image, sensor_msgs::PointCloud2>MySyncPolicy;
  message_filters::Synchronizer<MySyncPolicy> sync(MySyncPolicy(150), camera_sub, lidar_sub);
  sync.registerCallback(boost::bind(&RsCamFusion::callback, &fusion, _1, _2));

  fused_image_pub = nh.advertise<sensor_msgs::Image>("fused_image", 150);
  colored_cloud_showpub = nh.advertise<sensor_msgs::PointCloud2>("colored_cloud_toshow", 150);
  cloudmap = nh.advertise<sensor_msgs::PointCloud2>("cloudmap", 50);
  ros::spin();
  return 0;

}


