#include <cuda_functions.cuh>
#include <timer.h>
// #define MAX_ANGLE_COS 0.5
// #define MAX_ANGLE_COS 0.2588
#define MAX_ANGLE_COS 0.1
#define HUBER_RANGE 0.5
namespace cuda_function
{
    __global__ void sp_check(
        DeviceImage<int> *index_map_ptr,
        DeviceImage<uchar> *original_ptr,
        DeviceImage<uchar> *mask_ptr,
        DeviceImage<superpixel> *sp_map_ptr,
        DeviceImage<float4> *sp_average_ptr,
        DeviceImage<uchar> *temp_ptr
    )
    {
        const int x = blockIdx.x * blockDim.x + threadIdx.x;
        const int y = blockIdx.y * blockDim.y + threadIdx.y;
        const int width = index_map_ptr->width;
        const int height = index_map_ptr->height;
        const int sp_width = sp_map_ptr->width;
        const int sp_height = sp_map_ptr->height;
        if(x >= width - 2 || y >= height - 2)
            return;
        const int sp_index = index_map_ptr->atXY(x,y);
        const int sp_x = sp_index % sp_width;
        const int sp_y = sp_index / sp_width;
        if(sp_x >= sp_width || sp_y >= sp_height)
        {
            printf("(%d, %d) %d -> (%d, %d)\n", x, y, sp_index, sp_x, sp_y);
            // return;
        }
        const uchar mask_value = mask_ptr->atXY(sp_x, sp_y);
        uchar result = original_ptr->atXY(x,y);
        if(index_map_ptr->atXY(x+1,y) != sp_index || index_map_ptr->atXY(x,y+1) != sp_index)
            result=0;
        if(mask_value == 4)
            result=255;
        if(mask_value == 2)
            result=0;
        temp_ptr->atXY(x,y) = result;
    }

    void initialize_surfel_map_with_superpixel(
        cv::Mat &image,
        cv::Mat &depth,
        geometry_msgs::Pose &pose,
        vector<SurfelElement> &surfels,
        FuseParameters *parameter_ptr)
    {
        int width = image.cols;
        int height = image.rows;
        int sp_width = width / 8;
        int sp_height = height / 8;
        DeviceImage<uchar> device_image(width, height);
        device_image.setDevData(reinterpret_cast<uchar*>(image.data));
        DeviceImage<float> device_depth(width, height);
        device_depth.setDevData(reinterpret_cast<float*>(depth.data));
        DeviceImage<SurfelElement> device_surfels(sp_width, sp_height);
        device_surfels.zero();

        // superpixel related
        DeviceImage<int> sp_index_map(width, height);
        sp_index_map.zero();
        DeviceImage<superpixel> sp_map(sp_width, sp_height);
        DeviceImage<float4> sp_avg(sp_width, sp_height);
        initialize_superpixel(
            device_image,
            device_depth,
            parameter_ptr,
            sp_index_map,
            sp_map,
            sp_avg
        );

        SE3<float> se3_pose(
            pose.orientation.w,
            pose.orientation.x,
            pose.orientation.y,
            pose.orientation.z,
            pose.position.x,
            pose.position.y,
            pose.position.z);
        
        // initialize the surfels
        dim3 initialize_block;
        dim3 initialize_grid;
        initialize_block.x = 32;
        initialize_block.y = 32;
        initialize_grid.x = (sp_width + initialize_block.x - 1) / initialize_block.x;
        initialize_grid.y = (sp_height + initialize_block.y - 1) / initialize_block.y;
        DeviceImage<uchar> mask_map(sp_width, sp_height);
        mask_map.zero();
        initialize_surfels_kernel<<<initialize_grid, initialize_block>>>(
            sp_map.dev_ptr,
            sp_avg.dev_ptr,
            device_surfels.dev_ptr,
            se3_pose,
            parameter_ptr,
            0,
            mask_map.dev_ptr);
        if(surfels.size() != sp_width*sp_height)
            surfels.resize(sp_width*sp_height);
        device_surfels.getDevData(surfels.data());
    }

    void initialize_superpixel(
        DeviceImage<uchar> &device_image,
        DeviceImage<float> &device_depth,
        FuseParameters *parameter_ptr,
        DeviceImage<int> &device_index,
        DeviceImage<superpixel> &superpixel_map,
        DeviceImage<float4> &sp_avg)
    {
        printf("initialize superpixel!\n");
        // get necessary data to gpu
        int width = device_image.width;
        int height = device_image.height;

        // first construct the superpixel
        int sp_width = width / 8;
        int sp_height = height / 8;
        dim3 initialize_block;
        dim3 initialize_grid;
        initialize_block.x = 32;
        initialize_block.y = 32;
        initialize_grid.x = (sp_width + initialize_block.x - 1) / initialize_block.x;
        initialize_grid.y = (sp_height + initialize_block.y - 1) / initialize_block.y;
        initialize_superpixel_kernel<<<initialize_grid, initialize_block>>>(
            device_image.dev_ptr,
            device_depth.dev_ptr,
            superpixel_map.dev_ptr
        );
        cudaDeviceSynchronize();
        dim3 update_pixel_block;
        dim3 update_pixel_grid;
        update_pixel_block.x = 32;
        update_pixel_block.y = 32;
        update_pixel_grid.x = (width + update_pixel_block.x - 1) / update_pixel_block.x;
        update_pixel_grid.y = (height + update_pixel_block.y - 1) / update_pixel_block.y;
        dim3 update_seed_block;
        dim3 update_seed_grid;
        update_seed_block.x = 1;
        update_seed_block.y = 1;
        update_seed_block.z = 64;
        update_seed_grid.x = (sp_width + update_seed_block.x - 1) / update_seed_block.x;
        update_seed_grid.y = (sp_height + update_seed_block.y - 1) / update_seed_block.y;
        for(int update_i = 0; update_i < 5; update_i++)
        {
            update_superpixel_pixel<<<update_pixel_grid, update_pixel_block>>>(
                device_image.dev_ptr,
                device_depth.dev_ptr,
                device_index.dev_ptr,
                superpixel_map.dev_ptr
            );
            cudaDeviceSynchronize();
            update_superpixel_seed<<<update_seed_grid, update_seed_block>>>(
                device_image.dev_ptr,
                device_depth.dev_ptr,                
                device_index.dev_ptr,
                superpixel_map.dev_ptr
            );
            cudaDeviceSynchronize();
        }
        dim3 norm_block;
        dim3 norm_grid;
        norm_block.x = 32;
        norm_block.y = 32;
        norm_grid.x = (width + norm_block.x - 1) / norm_block.x;
        norm_grid.y = (height + norm_block.y - 1) / norm_block.y;
        DeviceImage<float3> norm_map(width, height);
        norm_map.zero();
        get_norm_map<<<norm_grid, norm_block>>>(norm_map.dev_ptr, device_depth.dev_ptr, parameter_ptr);
        sp_avg.zero();
        // sp_avg_kernel<<<update_seed_grid, update_seed_block>>>(
        //     device_depth.dev_ptr,
        //     norm_map.dev_ptr,
        //     device_index.dev_ptr,
        //     sp_avg.dev_ptr);
        dim3 robust_avg_block;
        dim3 robust_avg_grid;
        robust_avg_block.x = 256;
        robust_avg_block.y = 1;
        robust_avg_block.z = 1;
        robust_avg_grid.y = sp_width;
        robust_avg_grid.z = sp_height;
        sp_huber_avg_kernel<<<robust_avg_grid, robust_avg_block>>>(
            device_depth.dev_ptr,
            norm_map.dev_ptr,
            device_index.dev_ptr,
            sp_avg.dev_ptr);
        printf("initialize superpixel done!\n");        
    }

    void warp_local_map(
        geometry_msgs::Pose &warp_pose,
        vector<SurfelElement> &local_surfels)
    {
        const int local_surfels_num = local_surfels.size();
        dim3 fuse_local_block;
        fuse_local_block.x = 1024;
        dim3 fuse_local_grid;
        fuse_local_grid.x = (local_surfels_num + fuse_local_block.x - 1) / fuse_local_block.x;

        DeviceLinear<SurfelElement> device_local_surfels(local_surfels_num);
        device_local_surfels.setDevData(local_surfels.data());

        SE3<float> se3_warp_pose(
            warp_pose.orientation.w,
            warp_pose.orientation.x,
            warp_pose.orientation.y,
            warp_pose.orientation.z,
            warp_pose.position.x,
            warp_pose.position.y,
            warp_pose.position.z);
        warp_local_surfels_kernel<<<fuse_local_grid, fuse_local_block>>>(
            device_local_surfels.dev_ptr,
            se3_warp_pose
        );
        cudaDeviceSynchronize();
        device_local_surfels.getDevData(local_surfels.data());
    }

    void fuse_initialize_map(
        int reference_frame_index,
        cv::Mat &image,
        cv::Mat &depth,
        geometry_msgs::Pose &pose,
        vector<SurfelElement> &local_surfels,
        vector<SurfelElement> &new_surfels,
        FuseParameters *parameter_ptr)
    {
        printf("fuse_initialize_map begin!\n");
        const int width = image.cols;
        const int height = image.rows;
        const int local_surfels_num = local_surfels.size();
        const int sp_width = width / 8;
        const int sp_height = height / 8;
        dim3 fuse_local_block;
        fuse_local_block.x = 1024;
        dim3 fuse_local_grid;
        fuse_local_grid.x = (local_surfels_num + fuse_local_block.x - 1) / fuse_local_block.x;

        DeviceImage<uchar> device_image(width, height);
        device_image.setDevData(reinterpret_cast<uchar*>(image.data));
        DeviceImage<float> device_depth(width, height);
        device_depth.setDevData(reinterpret_cast<float*>(depth.data));
        DeviceLinear<SurfelElement> device_local_surfels(local_surfels_num);
        device_local_surfels.setDevData(local_surfels.data());

        // first get the superpixel map
        DeviceImage<int> sp_index_map(width, height);
        DeviceImage<superpixel> sp_map(sp_width, sp_height);
        DeviceImage<float4> sp_avg(sp_width, sp_height);
        initialize_superpixel(
            device_image,
            device_depth,
            parameter_ptr,
            sp_index_map,
            sp_map,
            sp_avg
        );

        SE3<float> se3_pose(
            pose.orientation.w,
            pose.orientation.x,
            pose.orientation.y,
            pose.orientation.z,
            pose.position.x,
            pose.position.y,
            pose.position.z);

        // fuse the surfels with the current frame
        DeviceImage<uchar> mask_map(sp_width, sp_height);
        mask_map.zero();
        fuse_local_surfels<<<fuse_local_grid, fuse_local_block>>>(
            sp_index_map.dev_ptr,
            sp_map.dev_ptr,
            sp_avg.dev_ptr,
            device_local_surfels.dev_ptr,
            mask_map.dev_ptr,
            reference_frame_index,
            se3_pose.inv(),
            se3_pose,
            parameter_ptr);
        remove_unstable_kernel<<<fuse_local_grid, fuse_local_block>>>(
            device_local_surfels.dev_ptr,
            reference_frame_index);
        device_local_surfels.getDevData(local_surfels.data());

        // initialize the surfels
        DeviceImage<SurfelElement> device_surfels(sp_width, sp_height);
        device_surfels.zero();
        dim3 initialize_block;
        dim3 initialize_grid;
        initialize_block.x = 32;
        initialize_block.y = 32;
        initialize_grid.x = (sp_width + initialize_block.x - 1) / initialize_block.x;
        initialize_grid.y = (sp_height + initialize_block.y - 1) / initialize_block.y;
        initialize_surfels_kernel<<<initialize_grid, initialize_block>>>(
            sp_map.dev_ptr,
            sp_avg.dev_ptr,
            device_surfels.dev_ptr,
            se3_pose,
            parameter_ptr,
            reference_frame_index,
            mask_map.dev_ptr);
        if(new_surfels.size() != sp_width*sp_height)
            new_surfels.resize(sp_width*sp_height);
        device_surfels.getDevData(new_surfels.data());

        // test the sp
        dim3 sp_block;
        dim3 sp_grid;
        sp_block.x = 16;
        sp_block.y = 16;
        sp_grid.x = (width + sp_block.x - 1) / sp_block.x;
        sp_grid.y = (height + sp_block.y - 1) / sp_block.y;
        DeviceImage<uchar> sp_test(width, height);
        sp_check<<<sp_grid, sp_block>>>(
            sp_index_map.dev_ptr,
            device_image.dev_ptr,
            mask_map.dev_ptr,
            sp_map.dev_ptr,
            sp_avg.dev_ptr,
            sp_test.dev_ptr
        );
        cudaDeviceSynchronize();
        cv::Mat test_img = cv::Mat(height, width, CV_8UC1);
        sp_test.getDevData(reinterpret_cast<uchar*>(test_img.data));
        cv::imshow("result", test_img);
        cv::waitKey(10);
        printf("fuse_initialize_map end\n");
    }

    // void warp_fuse_initialize_map(
    //     geometry_msgs::Pose &warp_pose,
    //     int this_frame_index,
    //     cv::Mat &image,
    //     cv::Mat &depth,
    //     geometry_msgs::Pose &pose,
    //     vector<SurfelElement> &local_surfels,
    //     vector<SurfelElement> &new_surfels,
    //     FuseParameters *parameter_ptr)
    // {
    //     const int width = image.cols;
    //     const int height = image.rows;
    //     const int local_surfels_num = local_surfels.size();
    //     const int sp_width = width / 8;
    //     const int sp_height = height / 8;
    //     dim3 fuse_local_block;
    //     fuse_local_block.x = 1024;
    //     dim3 fuse_local_grid;
    //     fuse_local_grid.x = (local_surfels_num + fuse_local_block.x - 1) / fuse_local_block.x;

    //     DeviceImage<uchar> device_image(width, height);
    //     device_image.setDevData(reinterpret_cast<uchar*>(image.data));
    //     DeviceImage<float> device_depth(width, height);
    //     device_depth.setDevData(reinterpret_cast<float*>(depth.data));
    //     DeviceLinear<SurfelElement> device_local_surfels(local_surfels_num);
    //     device_local_surfels.setDevData(local_surfels.data());
        
        
    //     SE3<float> se3_pose(
    //         pose.orientation.w,
    //         pose.orientation.x,
    //         pose.orientation.y,
    //         pose.orientation.z,
    //         pose.position.x,
    //         pose.position.y,
    //         pose.position.z);
        
    //     if( warp_pose.orientation.w != 1 || 
    //         warp_pose.position.x != 0 ||
    //         warp_pose.position.y != 0 ||
    //         warp_pose.position.z != 0 )
    //     {
    //         printf("cuda detect pose change !!!!\n warp local surfels according (%f, %f, %f, %f), (%f, %f, %f).\n",
    //             warp_pose.orientation.w,
    //             warp_pose.orientation.x,
    //             warp_pose.orientation.y,
    //             warp_pose.orientation.z,
    //             warp_pose.position.x,
    //             warp_pose.position.y,
    //             warp_pose.position.z);
    //         SE3<float> se3_warp_pose(
    //             warp_pose.orientation.w,
    //             warp_pose.orientation.x,
    //             warp_pose.orientation.y,
    //             warp_pose.orientation.z,
    //             warp_pose.position.x,
    //             warp_pose.position.y,
    //             warp_pose.position.z);
    //         warp_local_surfels_kernel<<<fuse_local_grid, fuse_local_block>>>(
    //             device_local_surfels.dev_ptr,
    //             se3_warp_pose
    //         );
    //     }

    //     // first get the superpixel map
    //     DeviceImage<int> sp_index_map(width, height);
    //     DeviceImage<superpixel> sp_map(sp_width, sp_height);
    //     DeviceImage<float4> sp_avg(sp_width, sp_height);
    //     initialize_superpixel(
    //         device_image,
    //         device_depth,
    //         parameter_ptr,
    //         sp_index_map,
    //         sp_map,
    //         sp_avg
    //     );

    //     // fuse the surfels with the current frame
    //     DeviceImage<uchar> mask_map(sp_width, sp_height);
    //     mask_map.zero();
    //     fuse_local_surfels<<<fuse_local_grid, fuse_local_block>>>(
    //         sp_index_map.dev_ptr,
    //         sp_map.dev_ptr,
    //         sp_avg.dev_ptr,
    //         device_local_surfels.dev_ptr,
    //         mask_map.dev_ptr,
    //         this_frame_index,
    //         se3_pose.inv(),
    //         se3_pose,
    //         parameter_ptr);
    //     remove_unstable_kernel<<<fuse_local_grid, fuse_local_block>>>(
    //         device_local_surfels.dev_ptr,
    //         this_frame_index);
    //     device_local_surfels.getDevData(local_surfels.data());

    //     // initialize the surfels
    //     DeviceImage<SurfelElement> device_surfels(sp_width, sp_height);
    //     device_surfels.zero();
    //     dim3 initialize_block;
    //     dim3 initialize_grid;
    //     initialize_block.x = 32;
    //     initialize_block.y = 32;
    //     initialize_grid.x = (sp_width + initialize_block.x - 1) / initialize_block.x;
    //     initialize_grid.y = (sp_height + initialize_block.y - 1) / initialize_block.y;
    //     initialize_surfels_kernel<<<initialize_grid, initialize_block>>>(
    //         sp_map.dev_ptr,
    //         sp_avg.dev_ptr,
    //         device_surfels.dev_ptr,
    //         se3_pose,
    //         parameter_ptr,
    //         this_frame_index,
    //         mask_map.dev_ptr);
    //     if(new_surfels.size() != sp_width*sp_height)
    //         new_surfels.resize(sp_width*sp_height);
    //     device_surfels.getDevData(new_surfels.data());

    //     // test the sp
    //     // dim3 sp_block;
    //     // dim3 sp_grid;
    //     // sp_block.x = 16;
    //     // sp_block.y = 16;
    //     // sp_grid.x = (width + sp_block.x - 1) / sp_block.x;
    //     // sp_grid.y = (height + sp_block.y - 1) / sp_block.y;
    //     // DeviceImage<uchar> sp_test(width, height);
    //     // sp_check<<<sp_grid, sp_block>>>(
    //     //     sp_index_map.dev_ptr,
    //     //     device_image.dev_ptr,
    //     //     mask_map.dev_ptr,
    //     //     sp_map.dev_ptr,
    //     //     sp_avg.dev_ptr,
    //     //     sp_test.dev_ptr
    //     // );
    //     // cudaDeviceSynchronize();
    //     // cv::Mat test_img = cv::Mat(height, width, CV_8UC1);
    //     // sp_test.getDevData(reinterpret_cast<uchar*>(test_img.data));
    //     // cv::imshow("result", test_img);
    //     // cv::waitKey(10);
    // }

    __global__ void warp_local_surfels_kernel(
        DeviceLinear<SurfelElement> *surfels_ptr,
        SE3<float> warp_pose)
    {
        const int x = blockIdx.x * blockDim.x + threadIdx.x;
        const int surfel_num = surfels_ptr->length;
        if(x >= surfel_num)
            return;
        SurfelElement my_element = surfels_ptr->at(x);
        float3 original_position = make_float3(my_element.px, my_element.py, my_element.pz);
        float3 original_norm = make_float3(my_element.nx, my_element.ny, my_element.nz);
        original_position = warp_pose * original_position;
        original_norm = warp_pose.rotate(original_norm);
        my_element.px = original_position.x;
        my_element.py = original_position.y;
        my_element.pz = original_position.z;
        my_element.nx = original_norm.x;
        my_element.ny = original_norm.y;
        my_element.nz = original_norm.z;
        surfels_ptr->at(x) = my_element;
    }

    __global__ void initialize_superpixel_kernel(
        DeviceImage<uchar> *image_ptr,
        DeviceImage<float> *depth_ptr,
        DeviceImage<superpixel> *super_pixel_ptr)
    {
        const int x = blockIdx.x * blockDim.x + threadIdx.x;
        const int y = blockIdx.y * blockDim.y + threadIdx.y;
        const int centre_x = x*8 + 4;
        const int centre_y = y*8 + 4;
        const int sp_width = super_pixel_ptr->width;
        const int sp_height = super_pixel_ptr->height;
        if(x >= sp_width || y >= sp_height)
            return;
        float intensity = image_ptr->atXY(centre_x, centre_y);
        float depth = depth_ptr->atXY(centre_x, centre_y);
        superpixel initialize_sp;
        initialize_sp.avg_intensity = intensity;
        initialize_sp.avg_depth = depth > 0 ? depth : 0;
        initialize_sp.x = centre_x;
        initialize_sp.y = centre_y;
        super_pixel_ptr->atXY(x,y) = initialize_sp;
    }

    __device__ bool get_sp_dist(
        const int &intensity, const float &depth, const int &x, const int &y, const superpixel &sp, float &without_depth, float &with_depth)
    {
        without_depth = 0;
        without_depth += (intensity - sp.avg_intensity)*(intensity - sp.avg_intensity) / 100.0;
        without_depth += ((x - sp.x)*(x - sp.x) + (y - sp.y)*(y - sp.y)) / 16.0;
        if(depth > 0 && sp.avg_depth > 0)
        {
            with_depth = without_depth;
            float inv_depth = 1.0/depth;
            float inv_sp_depth = 1.0 / sp.avg_depth;
            with_depth += (inv_depth - inv_sp_depth)*(inv_depth - inv_sp_depth) * 400.0; // x*400.0 = x/(0.05*0.05)
            // with_depth += (depth - sp.avg_depth)*(depth - sp.avg_depth) * 400.0; // x*400.0 = x/(0.05*0.05)
            return true;
        }
        else
            return false;
    }

    __global__ void update_superpixel_pixel(
        DeviceImage<uchar> *image_ptr,
        DeviceImage<float> *depth_ptr,
        DeviceImage<int> *assigned_index_ptr,
        DeviceImage<superpixel> *super_pixel_ptr)
    {
        const int x = blockIdx.x * blockDim.x + threadIdx.x;
        const int y = blockIdx.y * blockDim.y + threadIdx.y;
        const int image_width = image_ptr->width;
        const int image_height = image_ptr->height;
        const int sp_width = super_pixel_ptr->width;
        const int sp_height = super_pixel_ptr->height;
        const int intensity = image_ptr->atXY(x,y);
        const float depth = depth_ptr->atXY(x,y);
        if(x >= image_width || y >= image_height)
            return;
        const int sp_x = x / 8;
        const int sp_y = y / 8;
        int assigned_sp_index = 0;
        float min_dist = 1e6;
        superpixel temp_sp;
        float sp_cost[3][3];
        float sp_depth_cost[3][3];
        bool aggregate_w_depth = true;
        for(int scane_sp_i = -1; scane_sp_i <= 1; scane_sp_i++)
        for(int scane_sp_j = -1; scane_sp_j <= 1; scane_sp_j++)
        {
        {
            int check_sp_x = sp_x + scane_sp_i;
            int check_sp_y = sp_y + scane_sp_j;
            int dist_sp_x = abs(check_sp_x * 8 + 4 - x);
            int dist_sp_y = abs(check_sp_y * 8 + 4 - y);
            sp_cost[scane_sp_i+1][scane_sp_j+1] = 1e6;
            sp_depth_cost[scane_sp_i+1][scane_sp_j+1] = 1e6;
            if(dist_sp_x < 8 && dist_sp_y < 8
                && check_sp_x >= 0 && check_sp_x < sp_width && check_sp_y >= 0 && check_sp_y < image_height)
            {
                temp_sp = super_pixel_ptr->atXY(check_sp_x, check_sp_y);
                aggregate_w_depth &= get_sp_dist(intensity, depth, x, y, temp_sp, sp_cost[scane_sp_i+1][scane_sp_j+1], sp_depth_cost[scane_sp_i+1][scane_sp_j+1]);
            }
        }
        }
        for(int scane_sp_i = -1; scane_sp_i <= 1; scane_sp_i++)
        for(int scane_sp_j = -1; scane_sp_j <= 1; scane_sp_j++)
        {
        {
            if(aggregate_w_depth)
            {
                if(sp_depth_cost[scane_sp_i+1][scane_sp_j+1] < min_dist)
                {
                    min_dist = sp_depth_cost[scane_sp_i+1][scane_sp_j+1];
                    assigned_sp_index = sp_x + scane_sp_i + (sp_y + scane_sp_j) * sp_width;
                }
            }
            else
            {
                if(sp_cost[scane_sp_i+1][scane_sp_j+1] < min_dist)
                {
                    min_dist = sp_cost[scane_sp_i+1][scane_sp_j+1];
                    assigned_sp_index = sp_x + scane_sp_i + (sp_y + scane_sp_j) * sp_width;
                }
            }
        }
        }
        assigned_index_ptr->atXY(x,y) = assigned_sp_index;
    }

    __global__ void update_superpixel_seed(
        DeviceImage<uchar> *image_ptr,
        DeviceImage<float> *depth_ptr,
        DeviceImage<int> *assigned_index_ptr,
        DeviceImage<superpixel> *super_pixel_ptr)
    {
        const int sp_x = blockIdx.x * blockDim.x + threadIdx.x;
        const int sp_y = blockIdx.y * blockDim.y + threadIdx.y;
        const int image_width = image_ptr->width;
        const int image_height = image_ptr->height;
        const int sp_width = super_pixel_ptr->width;
        const int sp_height = super_pixel_ptr->height;
        superpixel prior_sp = super_pixel_ptr->atXY(sp_x, sp_y);
        if(sp_x >= sp_width || sp_y >= sp_height)
            return;
        const int sp_index = sp_width * sp_y + sp_x;
        const int patch_x = threadIdx.z % 8;
        const int patch_y = threadIdx.z / 8;
        __shared__ int patch_intensity[64];
        __shared__ float patch_depth[64];
        __shared__ float patch_dist[64];
        __shared__ int patch_location_x[64];
        __shared__ int patch_location_y[64];
        __shared__ uchar patch_indicator[64];
        __shared__ uchar patch_depth_indicator[64];
        int sum_intensity = 0;
        float sum_depth = 0;
        int sum_valid_depth = 0;
        int sum_pixel_num = 0;
        int sum_x = 0;
        int sum_y = 0;
        float max_dist = -1.0;;
        // scane the four 8x8 pathes
        // #pragma unroll
        for(int patch_i = 0; patch_i < 2; patch_i ++)
        {
        // #pragma unroll
        for(int patch_j = 0; patch_j < 2; patch_j ++)
        {
            int check_x = sp_x * 8 + patch_i * 8 + patch_x - 4;
            int check_y = sp_y * 8 + patch_j * 8 + patch_y - 4;
            patch_intensity[threadIdx.z] = 0;
            patch_depth[threadIdx.z] = 0;
            patch_indicator[threadIdx.z] = 0;
            patch_depth_indicator[threadIdx.z] = 0;
            patch_location_x[threadIdx.z] = 0;
            patch_location_y[threadIdx.z] = 0;
            patch_dist[threadIdx.z] = 0;
            if(check_x < image_width || check_y < image_height)
            {
                if(assigned_index_ptr->atXY(check_x, check_y) == sp_index)
                {
                    patch_indicator[threadIdx.z] = 1;
                    patch_intensity[threadIdx.z] = image_ptr->atXY(check_x, check_y);
                    patch_location_x[threadIdx.z] = check_x;
                    patch_location_y[threadIdx.z] = check_y;
                    patch_dist[threadIdx.z] = length(make_float2(check_x - prior_sp.x, check_y - prior_sp.y));
                    float my_depth = depth_ptr->atXY(check_x, check_y);
                    patch_depth[threadIdx.z] = my_depth > 0 ? my_depth : 0;
                    patch_depth_indicator[threadIdx.z] = my_depth > 0 ? 1 : 0;
                }
            }
            __syncthreads();
            // sum data
            for(int i = 32; i > 0; i = i / 2)
            {
                if(threadIdx.z < i)
                {
                    patch_indicator[threadIdx.z] += patch_indicator[threadIdx.z + i];
                    patch_depth_indicator[threadIdx.z] += patch_depth_indicator[threadIdx.z + i];
                    patch_intensity[threadIdx.z] += patch_intensity[threadIdx.z + i];
                    patch_depth[threadIdx.z] += patch_depth[threadIdx.z + i];
                    patch_location_x[threadIdx.z] += patch_location_x[threadIdx.z + i];
                    patch_location_y[threadIdx.z] += patch_location_y[threadIdx.z + i];
                    if(patch_dist[threadIdx.z + i] > patch_dist[threadIdx.z])
                        patch_dist[threadIdx.z] = patch_dist[threadIdx.z + i];
                }
                __syncthreads();
            }
            sum_pixel_num += patch_indicator[0];
            sum_intensity += patch_intensity[0];
            sum_depth += patch_depth[0];
            sum_valid_depth += patch_depth_indicator[0];
            sum_x += patch_location_x[0];
            sum_y += patch_location_y[0];
            max_dist = max_dist < patch_dist[0] ? patch_dist[0] : max_dist;
            __syncthreads();
        }
        }
        if(threadIdx.z == 0)
        {
            superpixel updated_sp;
            updated_sp.avg_intensity = float(sum_intensity) / sum_pixel_num;
            if(sum_valid_depth > 16)
                updated_sp.avg_depth = sum_depth / sum_valid_depth;
            else
                updated_sp.avg_depth = 0.0;
            updated_sp.x = float(sum_x) / sum_pixel_num;
            updated_sp.y = float(sum_y) / sum_pixel_num;
            updated_sp.size = max_dist;
            super_pixel_ptr->atXY(sp_x, sp_y) = updated_sp;
        }
    }

    __global__ void get_norm_map(
        DeviceImage<float3> *norm_ptr,
        DeviceImage<float> *depth_ptr,
        FuseParameters* parameter_ptr
    )
    {
        const int x = blockIdx.x * blockDim.x + threadIdx.x;
        const int y = blockIdx.y * blockDim.y + threadIdx.y;
        const int width = depth_ptr->width;
        const int height = depth_ptr->height;
        FuseParameters param = *parameter_ptr;
        if(x >= width - 1 || y >= height - 1)
            return;
        float my_depth, below_depth, right_depth;
        my_depth = depth_ptr->atXY(x,y);
        right_depth = depth_ptr->atXY(x+1,y);
        below_depth = depth_ptr->atXY(x,y+1);
        if(my_depth <= 0.0 || right_depth <= 0.0 || below_depth <= 0.0)
            return;

        float3 my_point = backproject(
            x, y, my_depth,
            param.fx, param.fy, param.cx, param.cy);
        float3 x_diff = backproject(
            x+1, y, right_depth,
            param.fx, param.fy, param.cx, param.cy) - my_point;
        float3 y_diff = backproject(
            x, y+1, below_depth,
            param.fx, param.fy, param.cx, param.cy) - my_point;
        float3 cross_result = cross(x_diff, y_diff);
        float3 surface_norm = normalize(cross_result);
        float view_angle_cos = dot(surface_norm, normalize(my_point));
        if (view_angle_cos < MAX_ANGLE_COS && view_angle_cos > - MAX_ANGLE_COS)
            return;
        norm_ptr->atXY(x,y) = surface_norm;
    }


    __global__ void sp_avg_kernel(
        DeviceImage<float> *depth_ptr,
        DeviceImage<float3> *norm_ptr,
        DeviceImage<int> *assigned_index_ptr,
        DeviceImage<float4> *result_ptr
    )
    {
        const int sp_x = blockIdx.x * blockDim.x + threadIdx.x;
        const int sp_y = blockIdx.y * blockDim.y + threadIdx.y;
        const int sp_width = result_ptr->width;
        const int sp_height = result_ptr->height;
        const int image_width = depth_ptr->width;
        const int image_height = depth_ptr->height;
        if(sp_x >= sp_width || sp_y >= sp_height)
            return;
        const int sp_index = sp_width * sp_y + sp_x;
        const int patch_x = threadIdx.z % 8;
        const int patch_y = threadIdx.z / 8;
        __shared__ float3 patch_norm[64];
        __shared__ float patch_depth[64];
        __shared__ int depth_indicator[64];
        float3 sum_norm = make_float3(0,0,0);
        float sum_depth = 0;
        int sum_valid_depth = 0;
        float temp_depth;
        float3 temp_norm;
        for(int patch_i = 0; patch_i < 2; patch_i ++)
        {
        for(int patch_j = 0; patch_j < 2; patch_j ++)
        {
            // load data
            int check_x = sp_x * 8 + patch_x + patch_i * 8 - 4;
            int check_y = sp_y * 8 + patch_y + patch_j * 8 - 4;
            patch_norm[threadIdx.z] = make_float3(0,0,0);
            patch_depth[threadIdx.z] = 0;
            depth_indicator[threadIdx.z] = 0;
            if(check_x >=0 && check_y >= 0 && check_x < image_width && check_y < image_height)
            {
                if(assigned_index_ptr->atXY(check_x, check_y) == sp_index)
                {
                    temp_norm = norm_ptr->atXY(check_x, check_y);
                    temp_depth = depth_ptr->atXY(check_x, check_y);
                    if(temp_norm.x != 0 || temp_norm.y != 0 || temp_norm.z != 0)
                    {
                        patch_norm[threadIdx.z] = temp_norm;
                        depth_indicator[threadIdx.z] = 1;
                        patch_depth[threadIdx.z] = temp_depth;
                    }
                }
            }
            __syncthreads();
            // sum data
            for(int i = 32; i > 0; i = i / 2)
            {
                if(threadIdx.z < i)
                {
                    patch_norm[threadIdx.z] += patch_norm[threadIdx.z + i];
                    patch_depth[threadIdx.z] += patch_depth[threadIdx.z + i];
                    depth_indicator[threadIdx.z] += depth_indicator[threadIdx.z + i];
                }
                __syncthreads();
            }
            sum_norm += patch_norm[0];
            sum_depth += patch_depth[0];
            sum_valid_depth += depth_indicator[0];
            __syncthreads();
        }
        }
        if(threadIdx.z == 0)
        {
            float length_norm = length(sum_norm);
            if(length_norm > 1.0 && sum_valid_depth > 32)
            {
                sum_norm = sum_norm / length_norm;
                sum_depth = sum_depth / (float) sum_valid_depth;
                result_ptr->atXY(sp_x, sp_y) = make_float4(sum_norm.x, sum_norm.y, sum_norm.z, sum_depth);
            }
        }
    }

    __global__ void sp_huber_avg_kernel(
        DeviceImage<float> *depth_ptr,
        DeviceImage<float3> *norm_ptr,
        DeviceImage<int> *assigned_index_ptr,
        DeviceImage<float4> *result_ptr
    )
    {
        // since cuda only support 64 threads on z dim of each block
        // here x indicates the thread for each patch and y,z indicate the superpixel
        const int patch_index = threadIdx.x;
        const int sp_x = blockIdx.y;
        const int sp_y = blockIdx.z;
        const int sp_width = result_ptr->width;
        const int sp_height = result_ptr->height;
        const int image_width = depth_ptr->width;
        const int image_height = depth_ptr->height;
        if(sp_x >= sp_width || sp_y >= sp_height)
            return;
        const int sp_index = sp_width * sp_y + sp_x;
        const int patch_x = patch_index % 16;
        const int patch_y = patch_index / 16;
        __shared__ float3 patch_norm[256];
        __shared__ float temp_a[256];
        __shared__ float temp_b[256];
        float my_depth = 0;
        float3 my_norm;
        float sum_valid_depth;
        bool vaild_point = false;
        // load data
        int check_x = sp_x * 8 + patch_x - 8;
        int check_y = sp_y * 8 + patch_y - 8;
        patch_norm[patch_index] = make_float3(0,0,0);
        temp_a[patch_index] = 0;
        temp_b[patch_index] = 0;
        if(check_x >=0 && check_y >= 0 && check_x < image_width && check_y < image_height)
        {
            if(assigned_index_ptr->atXY(check_x, check_y) == sp_index)
            {
                my_norm = norm_ptr->atXY(check_x, check_y);
                my_depth = depth_ptr->atXY(check_x, check_y);
                if(my_norm.x != 0 || my_norm.y != 0 || my_norm.z != 0)
                {
                    vaild_point = true;
                    temp_a[patch_index] = my_depth;
                    temp_b[patch_index] = 1.0;
                }
            }
        }

        __syncthreads();
        for(int i = 128; i > 0; i = i / 2)
        {
            if(patch_index < i)
            {
                temp_a[patch_index] += temp_a[patch_index + i];
                temp_b[patch_index] += temp_b[patch_index + i];
            }
            __syncthreads();
        }
        sum_valid_depth = temp_b[0];
        float mean_depth = temp_a[0] / temp_b[0];
        if(sum_valid_depth < 16)
            return;
        __syncthreads();
        // update the mean with huber norm using Newton's method
        for(int newton_i = 0; newton_i < 5; newton_i++)
        {
            // update the first and second gradient
            temp_a[patch_index] = 0;
            temp_b[patch_index] = 0;
            if(vaild_point)
            {
                float residual = mean_depth - my_depth;
                if(residual < HUBER_RANGE && residual > - HUBER_RANGE)
                {
                    temp_a[patch_index] = 2 * residual;
                    temp_b[patch_index] = 2;
                }
                else
                {
                    temp_a[patch_index] = residual > 0 ? 1 : -1;
                    temp_b[patch_index] = 0;
                }
            }
            __syncthreads();
            // aggregate
            for(int i = 128; i > 0; i = i / 2)
            {
                if(patch_index < i)
                {
                    temp_a[patch_index] += temp_a[patch_index + i];
                    temp_b[patch_index] += temp_b[patch_index + i];
                }
                __syncthreads();
            }
            mean_depth = mean_depth - temp_a[0] / (temp_b[0] + 10.0);
            __syncthreads();
        }

        // get the average norm of only inlier pixel
        if(mean_depth - my_depth > -HUBER_RANGE && mean_depth - my_depth < HUBER_RANGE && vaild_point)
        {
            patch_norm[patch_index] = my_norm;
        }
        __syncthreads();
        // aggregate
        for(int i = 128; i > 0; i = i / 2)
        {
            if(patch_index < i)
            {
                patch_norm[patch_index] += patch_norm[patch_index + i];
            }
            __syncthreads();
        }
        if(patch_index == 0)
        {
            float3 sum_norm = patch_norm[0];
            float length_norm = length(sum_norm);
            sum_norm = sum_norm / length_norm;
            result_ptr->atXY(sp_x, sp_y) = make_float4(sum_norm.x, sum_norm.y, sum_norm.z, mean_depth);
        }
    }

     __global__ void fuse_local_surfels(
        DeviceImage<int> *sp_index_map,
        DeviceImage<superpixel> *sp_map,
        DeviceImage<float4> *sp_avg_map,
        DeviceLinear<SurfelElement> *surfels_ptr,
        DeviceImage<uchar> *mask_ptr,
        int this_frame_index,
        SE3<float> world_in_cam,
        SE3<float> cam_in_world,
        FuseParameters *parameter_ptr)
    {
        const int x = blockIdx.x * blockDim.x + threadIdx.x;
        const int surfel_num = surfels_ptr->length;
        if(x >= surfel_num)
            return;

        FuseParameters param = *parameter_ptr;
        const int sp_width = param.width / 8;
        const int sp_height = param.height / 8;
        SurfelElement this_element = surfels_ptr->at(x);
        if(this_element.update_times == 0)
            return;

        float3 surfel_w_location = make_float3(this_element.px, this_element.py, this_element.pz);
        float3 surfel_c_location = world_in_cam * surfel_w_location;
        if(surfel_c_location.z <= param.near_dist)
            return;

        // project
        int p_u, p_v;
        project(surfel_c_location, param.fx, param.fy, param.cx, param.cy, p_u, p_v);
        if(p_u <= 1 || p_u >= param.width - 2 || p_v <= 1 || p_v >= param.height - 2)
            return;
        int sp_index = sp_index_map->atXY(p_u, p_v);
        int sp_x = sp_index % sp_width;
        int sp_y = sp_index / sp_width;
        if(sp_x >= sp_width || sp_y >= sp_height)
            return;

        // get spuerpixel data
        superpixel projected_sp = sp_map->atXY(sp_x, sp_y);
        float4 sp_avg_data = sp_avg_map->atXY(sp_x, sp_y);

        if(sp_avg_data.w < param.near_dist || sp_avg_data.w > param.far_dist)
        {
            mask_ptr->atXY(sp_x, sp_y) = 2;
            return;
        }
        
        float3 center_point = backproject(
            projected_sp.x, projected_sp.y, sp_avg_data.w,
            param.fx, param.fy, param.cx, param.cy);
        float3 norm_c = make_float3(sp_avg_data.x, sp_avg_data.y, sp_avg_data.z);
        float view_angle_cos = dot(norm_c, normalize(center_point));
        center_point = cam_in_world * center_point;
        float3 norm_w = cam_in_world.rotate(norm_c);

        float camera_f = (fabs(param.fx) + fabs(param.fy)) / 2.0;
        // tolerate_diff = depth * depth / ( baseline * focal_length ) * average_error_pixel
        float tolerate_diff = surfel_c_location.z * surfel_c_location.z / (0.5 * camera_f) * 2.5;
        // float tolerate_diff = surfel_c_location.z * surfel_c_location.z / (0.1 * camera_f) * 3.0;
        tolerate_diff = max(0.05, tolerate_diff);

        // associate the data
        if (fabs(sp_avg_data.w - surfel_c_location.z) < tolerate_diff &&
            (norm_w.x*this_element.nx+norm_w.y*this_element.ny+norm_w.z*this_element.nz) > 0.50)
        {
            float new_weight = get_weight(sp_avg_data.w);
            this_element.px = (this_element.px*this_element.weight+new_weight*center_point.x)/(this_element.weight+new_weight);
            this_element.py = (this_element.py*this_element.weight+new_weight*center_point.y)/(this_element.weight+new_weight);
            this_element.pz = (this_element.pz*this_element.weight+new_weight*center_point.z)/(this_element.weight+new_weight);
            float3 new_norm = normalize(
                make_float3(
                    (this_element.nx*this_element.weight+new_weight*norm_w.x)/(this_element.weight+new_weight),
                    (this_element.ny*this_element.weight+new_weight*norm_w.y)/(this_element.weight+new_weight),
                    (this_element.nz*this_element.weight+new_weight*norm_w.z)/(this_element.weight+new_weight)
                )
            );
            this_element.nx = new_norm.x;
            this_element.ny = new_norm.y;
            this_element.nz = new_norm.z;
            this_element.color = projected_sp.avg_intensity;

            float new_size = projected_sp.size * fabs(sp_avg_data.w / (camera_f * view_angle_cos * 1.414));
            this_element.size = new_size < this_element.size ? new_size : this_element.size;
            this_element.weight += new_weight;
            this_element.last_update = this_frame_index;
            if(this_element.update_times < 20)
                this_element.update_times += 1;
            surfels_ptr->at(x) = this_element;
            mask_ptr->atXY(sp_x, sp_y) = 1;
        }
        else
        {
            if(sp_avg_data.w > surfel_c_location.z)
            {
                this_element.update_times = 0;
                surfels_ptr->at(x) = this_element;
            }
            mask_ptr->atXY(sp_x, sp_y) = 4;
        }
    }
    
    __device__
    float3 backproject(const int &u, const int &v, float &depth, float &fx, float &fy, float &cx, float &cy)
    {
        return make_float3((u - cx) / fx * depth, (v - cy) / fy * depth, depth);
    }

    __device__
    void project(const float3 &xyz, 
        const float &fx, const float &fy, const float &cx, const float &cy,
        int &u, int &v)
    {
        u = int(xyz.x*fx/xyz.z+cx+0.5);
        v = int(xyz.y*fy/xyz.z+cy+0.5);
    }

    __device__
    float get_weight(float &depth)
    {
        return fminf(1.0/depth/depth,1.0);
    }

    __global__ void initialize_surfels_kernel(
        DeviceImage<superpixel> *sp_map_ptr,
        DeviceImage<float4> *avg_ptr,
        DeviceImage<SurfelElement> *surfels_ptr,
        SE3<float> cam_pose,
        FuseParameters *parameter_ptr,
        int this_frame_index,
        DeviceImage<uchar> *mask_ptr)
    {
        const int sp_x = blockIdx.x * blockDim.x + threadIdx.x;
        const int sp_y = blockIdx.y * blockDim.y + threadIdx.y;
        const int sp_width = sp_map_ptr->width;
        const int sp_height = sp_map_ptr->height;
        if(sp_x >= sp_width || sp_y >= sp_height)
            return;
        FuseParameters param = *parameter_ptr;
        superpixel this_sp = sp_map_ptr->atXY(sp_x, sp_y);
        float4 this_sp_data = avg_ptr->atXY(sp_x, sp_y);
        int mask_value = mask_ptr->atXY(sp_x, sp_y);
        if(mask_value == 1)
            return;
        if (this_sp_data.w <= param.near_dist || this_sp_data.w >= param.far_dist)
            return;

        float3 sp_point = backproject(
            this_sp.x, this_sp.y, this_sp_data.w,
            param.fx, param.fy, param.cx, param.cy);
        float3 norm_camera = make_float3(this_sp_data.x, this_sp_data.y, this_sp_data.z);
        float view_angle_cos = dot(norm_camera, normalize(sp_point));
        if (view_angle_cos < MAX_ANGLE_COS && view_angle_cos > - MAX_ANGLE_COS)
            return;

        float3 surface_norm = cam_pose.rotate(norm_camera);
        float3 point_world = cam_pose * sp_point;
        SurfelElement this_element;
        this_element.px = point_world.x;
        this_element.py = point_world.y;
        this_element.pz = point_world.z;
        this_element.nx = surface_norm.x;
        this_element.ny = surface_norm.y;
        this_element.nz = surface_norm.z;
        float camera_f = (fabs(param.fx) + fabs(param.fy)) / 2.0;
        this_element.size = this_sp.size * fabs(this_sp_data.w / (camera_f * view_angle_cos * 1.414));
        this_element.color = this_sp.avg_intensity;
        this_element.weight = get_weight(this_sp_data.w);
        this_element.update_times = 1;
        this_element.last_update = this_frame_index;
        surfels_ptr->atXY(sp_x,sp_y) = this_element;
    }

    __global__ void remove_unstable_kernel(
        DeviceLinear<SurfelElement> *surfels_ptr,
        int this_frame_index)
    {
        const int x = blockIdx.x * blockDim.x + threadIdx.x;
        int surfel_num = surfels_ptr->length;
        if(x >= surfel_num)
            return;

        SurfelElement this_element = surfels_ptr->at(x);

        if(this_frame_index - this_element.last_update > 5 && this_element.update_times < 5)
        {
            (surfels_ptr->at(x)).update_times = 0;
        }
    }
}