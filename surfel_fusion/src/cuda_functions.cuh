#pragma once
#include <opencv2/opencv.hpp>
#include <elements.h>
#include <parameters.h>
#include <vector>
#include <device_image.cuh>
#include <device_linear.cuh>
#include <se3.cuh>

using namespace std;

#define MIN_SURFEL_SIZE 0.02

namespace cuda_function
{
    struct superpixel{
        float avg_intensity;
        float size;
        float avg_depth;
        float x, y;
    };
    void warp_local_map(
        geometry_msgs::Pose &warp_pose,
        vector<SurfelElement> &local_surfels);
    void fuse_initialize_map(
        int reference_frame_index,
        cv::Mat &image,
        cv::Mat &depth,
        geometry_msgs::Pose &pose,
        vector<SurfelElement> &local_surfels,
        vector<SurfelElement> &new_surfels,
        FuseParameters *parameter_ptr);
    // void warp_fuse_initialize_map(
    //     geometry_msgs::Pose &warp_pose,
    //     int this_frame_index,
    //     cv::Mat &image,
    //     cv::Mat &depth,
    //     geometry_msgs::Pose &pose,
    //     vector<SurfelElement> &local_surfels,
    //     vector<SurfelElement> &new_surfels,
    //     FuseParameters *parameter_ptr);
    // void initialize_surfel_map_with_superpixel(
    //     cv::Mat &image,
    //     cv::Mat &depth,
    //     geometry_msgs::Pose &pose,
    //     vector<SurfelElement> &surfels,
    //     FuseParameters *parameter_ptr);
    void initialize_superpixel(
        DeviceImage<uchar> &device_image,
        DeviceImage<float> &device_depth,
        FuseParameters *parameter_ptr,
        DeviceImage<int> &device_index,
        DeviceImage<superpixel> &superpixel_map,
        DeviceImage<float4> &sp_avg);

    __device__
    float3 backproject(const int &u, const int &v, float &depth, float &fx, float &fy, float &cx, float &cy);
     __device__
    void project(const float3 &xyz, 
        const float &fx, const float &fy, const float &cx, const float &cy,
        int &u, int &v);
    __device__
    float get_weight(float &depth);
    // here the size of super-pixels are 8x8
    __global__ void warp_local_surfels_kernel(
        DeviceLinear<SurfelElement> *surfels_ptr,
        SE3<float> warp_pose);
    __global__ void initialize_superpixel_kernel(
        DeviceImage<uchar> *image_ptr,
        DeviceImage<float> *depth_ptr,
        DeviceImage<superpixel> *super_pixel_ptr);
    __global__ void update_superpixel_pixel(
        DeviceImage<uchar> *image_ptr,
        DeviceImage<float> *depth_ptr,
        DeviceImage<int> *assigned_index_ptr,
        DeviceImage<superpixel> *super_pixel_ptr);
    __global__ void update_superpixel_seed(
        DeviceImage<uchar> *image_ptr,
        DeviceImage<float> *depth_ptr,
        DeviceImage<int> *assigned_index_ptr,
        DeviceImage<superpixel> *super_pixel_ptr);
    __global__ void initialize_surfels_kernel(
        DeviceImage<superpixel> *sp_map_ptr,
        DeviceImage<float4> *avg_ptr,
        DeviceImage<SurfelElement> *surfels_ptr,
        SE3<float> cam_pose,
        FuseParameters *parameter_ptr,
        int this_frame_index,
        DeviceImage<uchar> *mask_ptr);
    __global__ void get_norm_map(
        DeviceImage<float3> *norm_ptr,
        DeviceImage<float> *depth_ptr,
        FuseParameters* parameter_ptr);
    __global__ void sp_huber_avg_kernel(
        DeviceImage<float> *depth_ptr,
        DeviceImage<float3> *norm_ptr,
        DeviceImage<int> *assigned_index_ptr,
        DeviceImage<float4> *result_ptr);
    __global__ void sp_avg_kernel(
        DeviceImage<float> *depth_ptr,
        DeviceImage<float3> *norm_ptr,
        DeviceImage<int> *assigned_index_ptr,
        DeviceImage<float4> *result_ptr);
    __global__ void fuse_local_surfels(
        DeviceImage<int> *sp_index_map,
        DeviceImage<superpixel> *sp_map,
        DeviceImage<float4> *sp_avg_map,
        DeviceLinear<SurfelElement> *surfels_ptr,
        DeviceImage<uchar> *mask_ptr,
        int this_frame_index,
        SE3<float> world_in_cam,
        SE3<float> cam_in_world,
        FuseParameters *parameter_ptr);
    __global__ void remove_unstable_kernel(
        DeviceLinear<SurfelElement> *surfels_ptr,
        int this_level);
}