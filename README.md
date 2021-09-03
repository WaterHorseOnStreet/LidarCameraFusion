# LidarCameraFusion

colored_pointcloud package

Install: 
cd to the src folder of a ros workspace, git clone this package and catkin_make this workspace.

Preparation: 
calibrate your camera, and your lidar-camera system,
then write the intrinsic matrix, distortion coefficients and the extrinsic matrix to config/calib_result.yaml,
finally change the camera_topic and lidar_topic to fit your own system. 

Usage: 
<launch your camera and lidar nodes>
roslaunch colored_pointcloud colored_pointcloud161.launch 



below is a piece of result of kitti dataset:

![](https://github.com/WaterHorseOnStreet/LidarCameraFusion/blob/main/colored_pointcloud-master/img/webwxgetmsgimg.jpeg)

![](https://github.com/WaterHorseOnStreet/LidarCameraFusion/blob/main/colored_pointcloud-master/img/2.jpeg)

=======
This is a demo which colors the pointcloud using image information. Using this tool we can create a coloured 3D world. But because of the sparsity of pointcloud comparing with image pixels, we need to research more on deepth completion later.


