#include "slic.h"
#include <iostream>
#include <thread>
#include <cmath>
#include <Eigen/Eigen>
#include <Eigen/Eigenvalues>

void SLIC::initialize_seeds(std::vector<Superpixel_seed> &superpixel_seeds)
{
    std::vector<std::thread> thread_pool;
    for (int i = 0; i < THREAD_NUM; i++)
    {
        std::thread this_thread(&SLIC::initialize_seeds_kernel, this, i, THREAD_NUM, superpixel_seeds);
        thread_pool.push_back(std::move(this_thread));
    }
    for (int i = 0; i < thread_pool.size(); i++)
        if (thread_pool[i].joinable())
            thread_pool[i].join();
}

void SLIC::initialize_seeds_kernel(int thread_i, int thread_num, std::vector<Superpixel_seed> &superpixel_seeds)
{
    int step = superpixel_seeds.size() / thread_num;
    int begin_index = step * thread_i;
    int end_index = begin_index + step;
    if(thread_i == thread_num - 1)
        end_index = superpixel_seeds.size();
    for (int seed_i = begin_index; seed_i < end_index; seed_i ++)
    {
        int sp_x = seed_i % sp_width;
        int sp_y = seed_i / sp_width;
        int image_x = sp_x * SP_SIZE + SP_SIZE/2;
        int image_y = sp_y * SP_SIZE + SP_SIZE/2;
        image_x = image_x < (image_width - 1) ? image_x : (image_width - 1);
        image_y = image_y < (image_height - 1) ? image_y : (image_height - 1);
        Superpixel_seed this_sp;
        this_sp.x = image_x;
        this_sp.y = image_y;
        this_sp.mean_intensity = image.at<uchar>(image_y, image_x);
        this_sp.mean_depth = depth.at<float>(image_y, image_x);
        this_sp.fused = false;
        superpixel_seeds[seed_i] = this_sp;
    }
}

bool SLIC::calculate_cost(
    float &nodepth_cost, float &depth_cost,
    const float &pixel_intensity, const float &pixel_inverse_depth,
    const int &x, const int &y,
    const int &sp_x, const int &sp_y)
{
    int sp_index = sp_y * sp_width + sp_x;
    nodepth_cost = 0;
    float dist =
        (superpixel_seeds[sp_index].x - x) * (superpixel_seeds[sp_index].x - x)
        + (superpixel_seeds[sp_index].y - y) * (superpixel_seeds[sp_index].y - y);
    nodepth_cost += dist / ((SP_SIZE / 2) * (SP_SIZE/2));
    float intensity_diff = (superpixel_seeds[sp_index].mean_intensity - pixel_intensity);
    nodepth_cost += intensity_diff * intensity_diff / 100.0;
    if (superpixel_seeds[sp_index].mean_depth <= 0 || pixel_inverse_depth <= 0)
        return false;
    depth_cost = nodepth_cost;
    float inverse_depth_diff = 1.0 / superpixel_seeds[sp_index].mean_depth - pixel_inverse_depth;
    depth_cost += inverse_depth_diff * inverse_depth_diff * 400.0;
    return true;
}

void SLIC::calculate_norms(
    std::vector<Superpixel_seed> &superpixel_seeds,
    std::vector<int> &superpixel_index)
{
    std::vector<std::thread> thread_pool;
    for (int i = 0; i < THREAD_NUM; i++)
    {
        std::thread this_thread(&SLIC::calculate_spaces_kernel, this, i, THREAD_NUM);
        thread_pool.push_back(std::move(this_thread));
    }
    for (int i = 0; i < thread_pool.size(); i++)
        if (thread_pool[i].joinable())
            thread_pool[i].join();
    thread_pool.clear();

    for (int i = 0; i < THREAD_NUM; i++)
    {
        std::thread this_thread(&SLIC::calculate_sp_norms_kernel, this, i, THREAD_NUM);
        thread_pool.push_back(std::move(this_thread));
    }
    for (int i = 0; i < thread_pool.size(); i++)
        if (thread_pool[i].joinable())
            thread_pool[i].join();
    thread_pool.clear();
}

void SLIC::calculate_spaces_kernel(int thread_i, int thread_num)
{
    int step_row = image_height / thread_num;
    int start_row = step_row * thread_i;
    int end_row = start_row + step_row;
    if(thread_i == thread_num - 1)
        end_row = image_height;
    for(int row_i = start_row; row_i < end_row; row_i++)
    for(int col_i = 0; col_i < image_width; col_i++)
    {
        int my_index = row_i * image_width + col_i;
        float my_depth = depth.at<float>(row_i, col_i);
        double x,y,z;
        back_project(col_i, row_i, my_depth, x,y,z);
        space_map[my_index * 3] = x;
        space_map[my_index * 3 + 1] = y;
        space_map[my_index * 3 + 2] = z;
    }
}

void SLIC::calculate_sp_norms_kernel(int thread_i, int thread_num,
    std::vector<Superpixel_seed> &superpixel_seeds,
    std::vector<int> &superpixel_index)
{
    int step = superpixel_seeds.size() / thread_num;
    int begin_index = step * thread_i;
    int end_index = begin_index + step;
    if (thread_i == thread_num - 1)
        end_index = superpixel_seeds.size();
    for (int seed_i = begin_index; seed_i < end_index; seed_i++)
    {
        int sp_x = seed_i % sp_width;
        int sp_y = seed_i / sp_width;
        int check_x_begin = sp_x * SP_SIZE + SP_SIZE / 2 - SP_SIZE;
        int check_y_begin = sp_y * SP_SIZE + SP_SIZE / 2 - SP_SIZE;
        std::vector<double> space_points;
        double sum_x, sum_y, sum_z;
        int sum_num;
        sum_x = 0;
        sum_y = 0;
        sum_z = 0;
        sum_num = 0;
        for (int check_j = check_y_begin; check_j < (check_y_begin + SP_SIZE * 2); check_j++)
        {
            for (int check_i = check_x_begin; check_i < (check_x_begin + SP_SIZE * 2); check_i++)
            {
                int pixel_index = check_j * image_width + check_i;
                if (pixel_index < 0 || pixel_index >= superpixel_index.size())
                    continue;
                if (superpixel_index[pixel_index] == seed_i)
                {
                    if(space_map[pixel_index * 3 + 2] < 0.1)
                        continue;
                    space_points.push_back(space_map[pixel_index * 3]);
                    space_points.push_back(space_map[pixel_index * 3 + 1]);
                    space_points.push_back(space_map[pixel_index * 3 + 2]);
                    sum_x += space_map[pixel_index * 3];
                    sum_y += space_map[pixel_index * 3 + 1];
                    sum_z += space_map[pixel_index * 3 + 2];
                    sum_num += 1;
                }
            }
        }
        if (sum_num < 3)
        {
            superpixel_seeds[seed_i].mean_depth = 0.0;
            continue;
        }
        sum_x /= sum_num;
        sum_y /= sum_num;
        sum_z /= sum_num;
        Eigen::Matrix3d covariance_m = Eigen::Matrix3d::Zero();
        for (int i = 0; i < sum_num; i++)
        {
            double dx = space_points[i * 3] - sum_x;
            double dy = space_points[i * 3 + 1] - sum_y;
            double dz = space_points[i * 3 + 2] - sum_z;
            covariance_m(0, 0) += dx * dx;
            covariance_m(0, 1) += dx * dy;
            covariance_m(0, 2) += dx * dz;
            covariance_m(1, 0) += dy * dx;
            covariance_m(1, 1) += dy * dy;
            covariance_m(1, 2) += dy * dz;
            covariance_m(2, 0) += dz * dx;
            covariance_m(2, 1) += dz * dy;
            covariance_m(2, 2) += dz * dz;
        }
        Eigen::EigenSolver<Eigen::Matrix3d> cov_eigen(covariance_m);
        double minium_ev = 1e6;
        double n_x, n_y, n_z;
        for(int i = 0; i < 3; i++)
        {
            if (cov_eigen.eigenvalues().real()(i) < minium_ev)
            {
                minium_ev = cov_eigen.eigenvalues().real()(i);
                n_x = cov_eigen.eigenvectors().real()(0, i);
                n_y = cov_eigen.eigenvectors().real()(1, i);
                n_z = cov_eigen.eigenvectors().real()(2, i);
            }
        }
        // std::cout << "The eigenvalues of" << seed_i << " cov_eigen are: " << std::endl
        //      << cov_eigen.eigenvalues() << std::endl;
        // std::cout << "The eigenvector of" << seed_i << " is " << std::endl
        //           << cov_eigen.eigenvectors() << std::endl;
        // std::cout << "The mini eigenvalues of" << seed_i << " is: " << minium_ev << std::endl;
        // std::cout << "The eigen value of" << seed_i << " is: " << n_x <<","<< n_y << "," << n_z << std::endl;

        // flip the dir of the norm if needed
        float view_cos = -1.0 * (n_x * sum_x + n_y * sum_y + n_z * sum_z);
        if (view_cos < 0)
        {
            view_cos *= -1.0;
            n_x *= -1.0;
            n_y *= -1.0;
            n_z *= -1.0;
        }
        superpixel_seeds[seed_i].norm_x = n_x;
        superpixel_seeds[seed_i].norm_y = n_y;
        superpixel_seeds[seed_i].norm_z = n_z;
        superpixel_seeds[seed_i].posi_x = sum_x;
        superpixel_seeds[seed_i].posi_y = sum_y;
        superpixel_seeds[seed_i].posi_z = sum_z;
        superpixel_seeds[seed_i].mean_depth = sum_z;
        superpixel_seeds[seed_i].view_cos = view_cos;

        // std::cout << n_x << "," << n_y << "," << n_z << std::endl;

        // double sum_length = sum_norm_x * sum_norm_x + sum_norm_y * sum_norm_y + sum_norm_z * sum_norm_z;
        // sum_length = std::sqrt(sum_length);
        // double avg_n_x = sum_norm_x / sum_length;
        // double avg_n_y = sum_norm_y / sum_length;
        // double avg_n_z = sum_norm_z / sum_length;
        // superpixel_seeds[seed_i].norm_x = avg_n_x;
        // superpixel_seeds[seed_i].norm_y = avg_n_y;
        // superpixel_seeds[seed_i].norm_z = avg_n_z;

        //     double avg_x,
        //     avg_y, avg_z;
        // back_project(
        //     superpixel_seeds[seed_i].x, superpixel_seeds[seed_i].y, superpixel_seeds[seed_i].mean_depth,
        //     avg_x, avg_y, avg_z);
        // double avg_n_b = avg_n_x * avg_x + avg_n_y * avg_y + avg_n_z * avg_z;
        // superpixel_seeds

        // newton method to optimzie
        // for (int gn_i = 0; gn_i < GN_ITERATION_NUM; gn_i ++)
        // {
        //     void();
        // }
    }
}

void SLIC::update_pixels(
    std::vector<Superpixel_seed> &superpixel_seeds,
    std::vector<int> &superpixel_index)
{
    std::vector<std::thread> thread_pool;
    for (int i = 0; i < THREAD_NUM; i++)
    {
        std::thread this_thread(&SLIC::update_pixels_kernel, this, i, THREAD_NUM);
        thread_pool.push_back(std::move(this_thread));
    }
    for (int i = 0; i < thread_pool.size(); i++)
        if (thread_pool[i].joinable())
            thread_pool[i].join();
}

void SLIC::update_pixels_kernel(
    int thread_i, int thread_num,
    std::vector<Superpixel_seed> &superpixel_seeds,
    std::vector<int> &superpixel_index)
{
    int step_row = image_height / thread_num;
    int start_row = step_row * thread_i;
    int end_row = start_row + step_row;
    if(thread_i == thread_num - 1)
        end_row = image_height;
    for(int row_i = start_row; row_i < end_row; row_i++)
    for(int col_i = 0; col_i < image_width; col_i++)
    {
        float my_intensity = image.at<uchar>(row_i, col_i);
        float my_inv_depth = 0.0;
        if (depth.at<float>(row_i, col_i) > 0.01)
            my_inv_depth = 1.0 / depth.at<float>(row_i, col_i);
        int base_sp_x = col_i / SP_SIZE;
        int base_sp_y = row_i / SP_SIZE;
        float no_depth_cost[3][3];
        float depth_cost[3][3];
        bool all_has_depth = true;
        for(int check_i = -1; check_i <= 1; check_i ++)
        for(int check_j = -1; check_j <= 1; check_j ++)
        {
            no_depth_cost[check_i + 1][check_j + 1] = 1e9;
            depth_cost[check_i + 1][check_j + 1] = 1e9;
            int check_sp_x = base_sp_x + check_i;
            int check_sp_y = base_sp_y + check_j;
            int dist_sp_x = fabs(check_sp_x * SP_SIZE + SP_SIZE/2 - col_i);
            int dist_sp_y = fabs(check_sp_y * SP_SIZE + SP_SIZE/2 - row_i);
            if (dist_sp_x < SP_SIZE && dist_sp_y < SP_SIZE &&
                check_sp_x >= 0 && check_sp_x < sp_width &&
                check_sp_y >= 0 && check_sp_y < sp_height)
            {
                all_has_depth &= calculate_cost(
                    no_depth_cost[check_i + 1][check_j + 1],
                    depth_cost[check_i + 1][check_j + 1],
                    my_intensity, my_inv_depth,
                    col_i, row_i, check_sp_x, check_sp_y);
            }
        }
        float min_dist = 1e6;
        int min_sp_index = -1;
        for (int check_i = -1; check_i <= 1; check_i++)
        for (int check_j = -1; check_j <= 1; check_j++)
        {
            if(all_has_depth)
            {
                if (depth_cost[check_i + 1][check_j + 1] < min_dist)
                {
                    min_dist = depth_cost[check_i + 1][check_j + 1];
                    min_sp_index = (base_sp_y + check_j) * sp_width + base_sp_x + check_i;
                }
            }
            else
            {
                if (no_depth_cost[check_i + 1][check_j + 1] < min_dist)
                {
                    min_dist = no_depth_cost[check_i + 1][check_j + 1];
                    min_sp_index = (base_sp_y + check_j) * sp_width + base_sp_x + check_i;
                }
            }
        }
        superpixel_index[row_i * image_width + col_i] = min_sp_index;
    }
}

void SLIC::update_seeds(
    std::vector<Superpixel_seed> &superpixel_seeds,
    std::vector<int> &superpixel_index)
{
    std::vector<std::thread> thread_pool;
    for (int i = 0; i < THREAD_NUM; i++)
    {
        std::thread this_thread(&SLIC::update_seeds_kernel, this, i, THREAD_NUM);
        thread_pool.push_back(std::move(this_thread));
    }
    for (int i = 0; i < thread_pool.size(); i++)
        if (thread_pool[i].joinable())
            thread_pool[i].join();
}

void SLIC::update_seeds_kernel(
    int thread_i, int thread_num, bool get_size,
    std::vector<Superpixel_seed> &superpixel_seeds,
    std::vector<int> &superpixel_index)
{
    int step = superpixel_seeds.size() / thread_num;
    int begin_index = step * thread_i;
    int end_index = begin_index + step;
    if(thread_i == thread_num - 1)
        end_index = superpixel_seeds.size();
    for (int seed_i = begin_index; seed_i < end_index; seed_i++)
    {
        int sp_x = seed_i % sp_width;
        int sp_y = seed_i / sp_width;
        int check_x_begin = sp_x * SP_SIZE + SP_SIZE / 2 - SP_SIZE;
        int check_y_begin = sp_y * SP_SIZE + SP_SIZE / 2 - SP_SIZE;
        std::vector<float> x_vector;
        std::vector<float> y_vector;
        float sum_x = 0;
        float sum_y = 0;
        float sum_intensity = 0.0;
        float sum_intensity_num = 0.0;
        float sum_depth = 0.0;
        float sum_depth_num = 0.0;
        for (int check_j = check_y_begin; check_j < check_y_begin + SP_SIZE * 2; check_j++)
        for (int check_i = check_x_begin; check_i < check_x_begin + SP_SIZE * 2; check_i ++)
        {
            int pixel_index = check_j * image_width + check_i;
            if (superpixel_index[pixel_index] == seed_i)
            {
                if(get_size)
                {
                    x_vector.push_back(check_i);
                    y_vector.push_back(check_j);
                }
                sum_x += check_i;
                sum_y += check_j;
                sum_intensity_num += 1.0;
                sum_intensity += image.at<uchar>(check_j, check_i);
                if (depth.at<float>(check_j, check_i) > 0.1)
                {
                    sum_depth += depth.at<float>(check_j, check_i);
                    sum_depth_num += 1.0;
                }
            }
        }
        sum_intensity /= sum_intensity_num;
        sum_x /= sum_intensity_num;
        sum_y /= sum_intensity_num;
        superpixel_seeds[seed_i].mean_intensity = sum_intensity;
        superpixel_seeds[seed_i].x = sum_x;
        superpixel_seeds[seed_i].y = sum_y;
        if (sum_depth_num > 0)
        {
            superpixel_seeds[seed_i].mean_depth = sum_depth / sum_depth_num;
        }
        else
        {
            superpixel_seeds[seed_i].mean_depth = 0.0;
        }
        if (get_size)
        {
            float max_dist = 0;
            for(int pixel_i = 0; pixel_i < x_vector.size(); pixel_i++)
            {
                float this_dist = (x_vector[pixel_i] - sum_x) * (x_vector[pixel_i] - sum_x) 
                    + (y_vector[pixel_i] - sum_y) * (y_vector[pixel_i] - sum_y);
                if (this_dist > max_dist)
                    max_dist = this_dist;
            }
            superpixel_seeds[seed_i].size = std::sqrt(max_dist);
        }
    }
}

SLIC::initialize(int _width, int _height, float _fx, float _fy, float _cx, float _cy)
{
    width = _width;
    height = _height;
    sp_width = width / 8;
    sp_height = height / 8;
    fx = _fx;
    fy = _fy;
    cx = _cx;
    cy = _cy;
}

void generate(
    cv::Mat &input_image, cv::Mat &input_depth,
    std::vector<Superpixel_seed> &superpixel_seeds,
    std::vector<int> &superpixel_index)
{
    if(superpixel_seeds.size() != sp_width * sp_height)
    {
        superpixel_seeds.resize(sp_width * sp_height);
        superpixel_seeds.clear();
    }
    if(superpixel_index.size() != image_width * image_height)
    {
        superpixel_index.resize(image_width * image_height);
        superpixel_index.clear();
    }
    if (space_map.size() != image_width * image_height * 3)
    {
        space_map.resize(image_width * image_height * 3);
        space_map.clear();
    }

    image = input_image.clone();
    depth = input_depth.clone();

    initialize_seeds();
    for (int it_i = 0; it_i < ITERATION_NUM; it_i++)
    {
        update_pixels();
        update_seeds();
    }
    calculate_norms();
}

void SLIC::get_sp_map(cv::Mat &result)
{
    // average intensity
    result = cv::Mat(image_height, image_width, CV_8UC1);
    for (int i = 0; i < superpixel_index.size(); i++)
    {
        int p_x = i % image_width;
        int p_y = i / image_width;
        result.at<uchar>(p_y, p_x) = superpixel_seeds[superpixel_index[i]].mean_depth/5.0*255;
    }

    // depth segmentation
    // result = cv::Mat(image_height, image_width, CV_8UC1);
    // for (int i = 0; i < superpixel_index.size(); i++)
    // {
    //     int p_x = i % image_width;
    //     int p_y = i / image_width;
    //     result.at<uchar>(p_y, p_x) = image.at<uchar>(p_y, p_x);
    //     // result.at<uchar>(p_y, p_x) = depth.at<float>(p_y, p_x)/3.0*255.0;
    //     int my_index = superpixel_index[i];
    //     if (p_x + 1 < image_width && superpixel_index[i + 1] != my_index)
    //         result.at<uchar>(p_y, p_x) = 0;
    //     if (p_y + 1 < image_height && superpixel_index[i + image_width] != my_index)
    //         result.at<uchar>(p_y, p_x) = 0;
    // }

    // norm visualization
    // result = cv::Mat(image_height, image_width, CV_8UC3);
    // for(int j = 0; j < image_height; j++)
    // for(int i = 0; i < image_width; i++)
    // {
    //     int pixel_num = (j * image_width + i) * 3;
    //     cv::Vec3b this_norm;
    //     this_norm[0] = fabs(norm_map[pixel_num]) * 255;
    //     this_norm[1] = fabs(norm_map[pixel_num + 1]) * 255;
    //     this_norm[2] = fabs(norm_map[pixel_num + 2]) * 255;
    //     result.at<cv::Vec3b>(j, i) = this_norm;
    // }

    // sp norm visualization
    // result = cv::Mat(image_height, image_width, CV_8UC3);
    // for(int j = 0; j < image_height; j++)
    // for(int i = 0; i < image_width; i++)
    // {
    //     // pixel sp index
    //     int sp_index = superpixel_index[j*image_width + i];
    //     cv::Vec3b this_norm;
    //     this_norm[0] = fabs(superpixel_seeds[sp_index].norm_x) * 255;
    //     this_norm[1] = fabs(superpixel_seeds[sp_index].norm_y) * 255;
    //     this_norm[2] = fabs(superpixel_seeds[sp_index].norm_z) * 255;
    //     result.at<cv::Vec3b>(j, i) = this_norm;
    // }
    // for (int i = 0; i < superpixel_index.size(); i++)
    // {
    //     int p_x = i % image_width;
    //     int p_y = i / image_width;
    //     int my_index = superpixel_index[i];
    //     if (p_x + 1 < image_width && superpixel_index[i + 1] != my_index)
    //         result.at<cv::Vec3b>(p_y, p_x) = cv::Vec3b(0, 0, 0);
    //     if (p_y + 1 < image_height && superpixel_index[i + image_width] != my_index)
    //         result.at<cv::Vec3b>(p_y, p_x) = cv::Vec3b(0, 0, 0);
    // }
}