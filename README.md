# DenseSurfelMapping

**WARNING!**

**We have cleaned the code such that it can run without GPU acceleration. The code have not been fully tested after the refactoring. If you have any questions or suggestions, please let us know in the issue.**

## A depth map fusion method

This is a depth map fusion method following the ICRA 2019 submission **Real-time Scalable Dense Surfel Mapping**, Kaixuan Wang, Fei Gao, and Shaojie Shen.

Given a sequence of depth images, intensity images, and camera poses, the proposed methods can fuse them into a globally consistent model using surfel representation. The fusion method supports both [ORB-SLAM2](https://github.com/raulmur/ORB_SLAM2) and [VINS-Mono](https://github.com/HKUST-Aerial-Robotics/VINS-Mono) (a little modification is required) so that you can use it in RGB-D, stereo, or visual-inertial cases according to your setups. We develop the method based on the motivation that the fusion method: (1) can support loop closure (so that it can be consistent with other state-of-the-art SLAM methods),  (2) do not require much CPU/memory resources to reconstruct a fine model in real-time, (3) can be scaled to large environments. These requirements are of vital importance in robot navigation tasks that the robot can safly navigate in the environment with odometry-consistent dense maps.

An example to show the usage of the surfel mapping is shown below.

<p align="center">
<img src="fig/example.png" alt="mapping example" width = "623" height = "300">
</p>

Left is the overview of the environment, the middle is the reconstructed results (visualized as point clouds in rviz of ROS) of our method, and right is the result using [OpenChisel](https://github.com/personalrobotics/OpenChisel). We use VINS-Mono to track the camera motion with loop closure, and [MVDepthNet](https://github.com/HKUST-Aerial-Robotics/MVDepthNet) to estimate the depth maps. The black line is the path of the camera. In the reconstruction, loop closure is enabled to correct the detected drift. OpenChisel is a great project to reconstruct the environment using the truncated signed distance function (TSDF). However, as shown in the example, it is not suitable to be used with SLAM systems that have loop closure abilities.

The system can also be applied to the KITTI datasets in real-time with only CPU computation.

<p align="center">
<img src="fig/example2.png" alt="mapping example" width = "465" height = "300">
</p>

The top row is the reconstruction using stereo cameras and the bottom row is the reconstruction using **only the left camera**. Details can be found in the paper.

A video can be used to illustrate the performance of the system and how we apply it into an autonomous navigation:
<p align="center">
<a href="https://youtu.be/2gZNpFE_yI4" target="_blank"><img src="fig/cover.jpg" 
alt="video" width="432" height="316" border="10" /></a>
</p>

## Running with VINS-Mono

We have use the surfel fusion with VINS-Mono in lots of UAV projects. For depth estimation, we recommend high quality depth methods/devices, for example [MVDepthNet](https://github.com/HKUST-Aerial-Robotics/MVDepthNet) or intel-realsense. Please refer to ```/launch/fuse_depthnet.launch``` for detailed parameters. The system takes paired grey_image and depth map as input. Since VINS-Mono publishes imu poses, we also need to receive ```/vins_estimator/extrinsic``` for converting imu poses into camera poses.

## Generate Colored Map
<p align="center">
<img src="fig/colored_map.png" alt="mapping example">
</p>

We have use Intel Realsense with VINS-Fusion to generate a colored mesh map. Colored image and depth image(aligned to color image) are used to reconstruct the environment, while VINS-Fusion used stereo gray image to calculate VIO.

**Step1: Download and run dataset**

Download rosbag we provided [here](https://hkustconnect-my.sharepoint.com/:u:/g/personal/cliuci_connect_ust_hk/Ef-CaCA7ZZ1Lu9EHB84f314B631jJnot31kDbREENSns4Q?e=TFqdWs) and run the bag,
```js
rosbag play surfel_color02.bag
```
We have solved VIO from VINS. For detailed configuration about VINS, please refer to [VINS-Fusion](https://github.com/HKUST-Aerial-Robotics/VINS-Fusion).

**Step2: Launch Surfel Fusion**
```js
roslaunch surfel_fusion vins_realsense.launch
```

**Step 3: Save Map**

Send command to stop surfel and save map in .ply and .pcd
````js
rostopic pub /surfel_cmd std_msgs/Int16 "data: 2"  
````

## Ackonwledgement
We thank Gao Fei, Pan Jie, and Wang Luqi, for their contribution to the code and suggestions.