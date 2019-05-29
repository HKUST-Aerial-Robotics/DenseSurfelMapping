#pragma once
#include <opencv2/opencv.hpp>
#include <vector>
#include "elements.h"

#define ITERATION_NUM 3
#define GN_ITERATION_NUM 0
#define THREAD_NUM 20
#define SP_SIZE 10
class SLIC
{
private:
    int image_width;
    int image_height;
    int sp_width;
    int sp_height;

    cv::Mat image;
    cv::Mat depth;

    std::vector<double> space_map;
    std::vector<float> norm_map;
    
    float fx, fy, cx, cy;
    void back_project(
        const float &u, const float &v, const float &depth, double&x, double&y, double&z)
    {
        x = (u - cx) / fx * depth;
        y = (v - cy) / fy * depth;
        z = depth;
    }
    bool calculate_cost(
        float &nodepth_cost, float &depth_cost,
        const float &pixel_intensity, const float &pixel_inverse_depth,
        const int &x, const int &y,
        const int &sp_x, const int &sp_y);
    void update_pixels_kernel(
        int thread_i, int thread_num,
        std::vector<Superpixel_seed> &superpixel_seeds,
        std::vector<int> &superpixel_index);
    void update_pixels(
        std::vector<Superpixel_seed> &superpixel_seeds,
        std::vector<int> &superpixel_index);
    void update_seeds_kernel(
        int thread_i, int thread_num,
        std::vector<Superpixel_seed> &superpixel_seeds,
        std::vector<int> &superpixel_index);
    void update_seeds(
        std::vector<Superpixel_seed> &superpixel_seeds,
        std::vector<int> &superpixel_index);
    void initialize_seeds_kernel(
        int thread_i, int thread_num, bool get_size,
        std::vector<Superpixel_seed> &superpixel_seeds);
    void initialize_seeds(std::vector<Superpixel_seed> &superpixel_seeds);
    void calculate_spaces_kernel(int thread_i, int thread_num);
    void calculate_sp_norms_kernel(int thread_i, int thread_num,
        std::vector<Superpixel_seed> &superpixel_seeds,
        std::vector<int> &superpixel_index);
    void calculate_norms(
        std::vector<Superpixel_seed> &superpixel_seeds,
        std::vector<int> &superpixel_index);

  public:
    void initialize(int _width,int _height, float _fx, float _fy, float _cx, float _cy);
    void generate(
        cv::Mat &input_image, cv::Mat &input_depth,
        std::vector<Superpixel_seed> &superpixel_seeds,
        std::vector<int> &superpixel_index);
    void get_sp_map(cv::Mat &result);
};