#include <surfel_map.h>
// #include <cuda_functions.cuh>
#include <timer.h>
#include <algorithm>
#include <pcl/io/pcd_io.h>

SurfelMap::SurfelMap(ros::NodeHandle &_nh):
nh(_nh),
// fuse_param_gpuptr(NULL),
inactive_pointcloud(new PointCloud)
{
    // get the parameters
    bool get_all = true;
    get_all &= nh.getParam("cam_width", cam_width);
    get_all &= nh.getParam("cam_height", cam_height);
    get_all &= nh.getParam("cam_fx", cam_fx);
    get_all &= nh.getParam("cam_cx", cam_cx);
    get_all &= nh.getParam("cam_fy", cam_fy);
    get_all &= nh.getParam("cam_cy", cam_cy);

    camera_matrix = Eigen::Matrix3d::Zero();
    camera_matrix(0, 0) = cam_fx;
    camera_matrix(0, 2) = cam_cx;
    camera_matrix(1, 1) = cam_fy;
    camera_matrix(1, 2) = cam_cy;
    camera_matrix(2, 2) = 1.0;

    get_all &= nh.getParam("fuse_far_distence", far_dist);
    get_all &= nh.getParam("fuse_near_distence", near_dist);
    get_all &= nh.getParam("drift_free_poses", drift_free_poses);

    if(!get_all)
        printf("ERROR! Do not have enough parameters!");
    else
    {
        printf("Have the following settings: \n");
        printf("camera matrix: \n");
        cout << camera_matrix << endl;
        printf("fuse the distence between %4f m and %4f m.\n", near_dist, far_dist);
    }

    // fuse_param.fx = cam_fx;
    // fuse_param.fy = cam_fy;
    // fuse_param.cx = cam_cx;
    // fuse_param.cy = cam_cy;
    // fuse_param.width = cam_width;
    // fuse_param.height = cam_height;
    // fuse_param.far_dist = far_dist;
    // fuse_param.near_dist = near_dist;
    // // local_loop_warp = Eigen::Matrix4d::Identity();
    // cudaMalloc(&fuse_param_gpuptr, sizeof(FuseParameters));
    // cudaMemcpy(fuse_param_gpuptr, &fuse_param, sizeof(FuseParameters), cudaMemcpyHostToDevice);

    fusion_functions.initialize(cam_width, cam_height, cam_fx, cam_fy, cam_cx, cam_cy, far_dist, near_dist);

    // ros publisher
    pointcloud_publish = nh.advertise<PointCloud>("pointcloud", 10);
    set_pointcloud_publish = nh.advertise<PointCloud>("set_pointcloud", 10);
    cam_pose_publish = nh.advertise<geometry_msgs::PoseStamped>("cam_pose", 10);

    raw_pointcloud_publish = nh.advertise<PointCloud>("raw_pointcloud", 10);
    loop_path_publish = nh.advertise<nav_msgs::Path>("fusion_loop_path", 10);
    driftfree_path_publish = nh.advertise<visualization_msgs::Marker>("driftfree_loop_path", 10);
    loop_marker_publish = nh.advertise<visualization_msgs::Marker>("loop_marker", 10);

    // render_tool initialize
    // render_tool.initialize_rendertool(cam_width, cam_height, cam_fx, cam_fy, cam_cx, cam_cy);

    //
    is_first_path = true;
    extrinsic_matrix_initialized = false;
}

SurfelMap::~SurfelMap()
{
    // if (fuse_param_gpuptr)
    //     cudaFree(fuse_param_gpuptr);
}

void SurfelMap::save_map(const std_msgs::StringConstPtr &save_map_input)
{
    string save_name = save_map_input->data;
    printf("save mesh modelt to %s.\n", save_name.c_str());
    save_mesh(save_name);
    printf("save done!\n");
}

void SurfelMap::image_input(const sensor_msgs::ImageConstPtr &image_input)
{
    // printf("receive image!\n");
    cv_bridge::CvImagePtr image_ptr = cv_bridge::toCvCopy(image_input, sensor_msgs::image_encodings::TYPE_8UC1);
    cv::Mat image = image_ptr->image;
    ros::Time stamp = image_ptr->header.stamp;
    image_buffer.push_back(std::make_pair(stamp, image));
    synchronize_msgs();
}

void SurfelMap::depth_input(const sensor_msgs::ImageConstPtr &depth_input)
{
    {
        cv_bridge::CvImagePtr image_ptr;
        image_ptr = cv_bridge::toCvCopy(depth_input, depth_input->encoding);
        constexpr double kDepthScalingFactor = 0.001;
        if(depth_input->encoding == sensor_msgs::image_encodings::TYPE_16UC1)
            (image_ptr->image).convertTo(image_ptr->image, CV_32FC1, kDepthScalingFactor);
        // image_ptr = cv_bridge::toCvCopy(depth_input, sensor_msgs::image_encodings::TYPE_32FC1);
        cv::Mat image = image_ptr->image;
        ros::Time stamp = image_ptr->header.stamp;
        depth_buffer.push_back(std::make_pair(stamp, image));
        synchronize_msgs();
    }
}

void SurfelMap::synchronize_msgs()
{
    std::chrono::time_point<std::chrono::system_clock> start_time, end_time;
    std::chrono::duration<double> total_time;
    start_time = std::chrono::system_clock::now();

    if(pose_reference_buffer.size() == 0)
        return;
    
    for(int scan_pose = 0; scan_pose < pose_reference_buffer.size(); scan_pose++)
    {
        ros::Time fuse_stamp = pose_reference_buffer[scan_pose].first;
        double pose_reference_time = fuse_stamp.toSec();
        int image_num = -1;
        int depth_num = -1;
        for(int image_i = 0; image_i < image_buffer.size(); image_i++)
        {
            double image_time = image_buffer[image_i].first.toSec();
            if(fabs(image_time - pose_reference_time) < 0.01)
            {   
                /*if(fabs(image_time - pose_reference_time) > 0.00001)
                    ROS_ERROR("diff time:= %f", fabs(image_time - pose_reference_time));*/
                image_num = image_i;
            }
        }
        for(int depth_i = 0; depth_i < depth_buffer.size(); depth_i++)
        {
            double depth_time = depth_buffer[depth_i].first.toSec();
            if(fabs(depth_time - pose_reference_time) < 0.01)
            {   
                /*if(fabs(depth_time - pose_reference_time) > 0.00001)
                    ROS_ERROR("diff time:= %f", fabs(depth_time - pose_reference_time));*/
                depth_num = depth_i;
            }
        }

        if( image_num < 0 || depth_num < 0)
            continue;

        int relative_index = pose_reference_buffer[scan_pose].second;
        geometry_msgs::Pose fuse_pose = poses_database[relative_index].cam_pose;
        Eigen::Matrix4d fuse_pose_eigen;
        pose_ros2eigen(fuse_pose, fuse_pose_eigen);

        move_add_surfels(relative_index);

        // fuse the current image/depth
        // /rintf("fuse map begins!\n");

        //std::cout<<"image_num:"<<image_num<<"\nima"
        cv::Mat image, depth;
        image = image_buffer[image_num].second;
        depth = depth_buffer[depth_num].second;
        /*image = image_buffer.front().second;
        depth = depth_buffer.front().second;*/
        fuse_map(image, depth, fuse_pose_eigen.cast<float>(), relative_index);
        //printf("fuse map done!\n");

        move_all_surfels();

        for(int delete_pose = 0; delete_pose <= scan_pose; delete_pose ++)
            pose_reference_buffer.pop_front();
        for(int delete_image = 0; delete_image <= image_num; delete_image++)
            image_buffer.pop_front();
        for(int delete_depth = 0; delete_depth <= depth_num; delete_depth++)
            depth_buffer.pop_front();

        // {
        //     // debug print the pose value
        //     printf("print the pose value\n");
        //     for(int i = 0; i < poses_database.size(); i++)
        //     {
        //         printf("\nthe pose %d, pose: (%f, %f, %f, %f) position (%f, %f, %f)", i,
        //             poses_database[i].cam_pose.orientation.x,
        //             poses_database[i].cam_pose.orientation.y,
        //             poses_database[i].cam_pose.orientation.z,
        //             poses_database[i].cam_pose.orientation.w,
        //             poses_database[i].cam_pose.position.x,
        //             poses_database[i].cam_pose.position.y,
        //             poses_database[i].cam_pose.position.z
        //         );
        //     }
        //     printf("\n");
        // }
        
        end_time = std::chrono::system_clock::now();
        total_time = end_time - start_time;
        //printf("fuse surfels cost %f ms.\n", total_time.count()*1000.0);
        start_time = std::chrono::system_clock::now();    

        // publish results
        // publish_raw_pointcloud(depth, image, fuse_pose);
        // publish_neighbor_pointcloud(fuse_stamp, relative_index);
        publish_this_set_pointcloud(fuse_stamp, relative_index);
        publish_pose_graph(fuse_stamp, relative_index);
        // render_depth(fuse_pose_ros);
        // if(poses_database.size()%2==0)
        // {
        // publish_all_pointcloud(fuse_stamp);
        // }
        end_time = std::chrono::system_clock::now();
        total_time = end_time - start_time;
        // printf("publish results cost %f ms.\n", total_time.count()*1000.0);
        // calculate_memory_usage();

        // break;
    }

    /*ROS_WARN("pose_buffer size:  = %d", pose_reference_buffer.size());
    ROS_WARN("image_buffer size: = %d", image_buffer.size());
    ROS_WARN("depth_buffer size: = %d", depth_buffer.size());*/
}

void SurfelMap::extrinsic_input(const nav_msgs::OdometryConstPtr &ex_input)
{
    geometry_msgs::Pose ex_pose = ex_input->pose.pose;
    pose_ros2eigen(ex_pose, extrinsic_matrix);
    // std::cout << "receive extrinsic pose" << std::endl <<  extrinsic_matrix << std::endl;
    extrinsic_matrix_initialized = true;
}

void SurfelMap::loop_info_input(const cerebro::LoopEdgeConstPtr &loop_info_input)
{
    printf("we receive!\n");
    double previous_stamp = loop_info_input->timestamp0.toSec();
    double current_stamp = loop_info_input->timestamp1.toSec();
    linked_pose_stamps.push_back(std::make_pair(previous_stamp, current_stamp));

    for (int linked_i = 0; linked_i < linked_pose_stamps.size(); linked_i++)
    {
        previous_stamp = linked_pose_stamps[linked_i].first;
        current_stamp = linked_pose_stamps[linked_i].second;
        if (previous_stamp < 0 || current_stamp < 0)
        {
            continue;
        }
        printf("The loop link %f -> %f\n", previous_stamp, current_stamp);
        int previouse_idx = -1;
        int current_idx = -1;
        for (int i = 0; i < poses_database.size(); i++)
        {
            if (previouse_idx < 0 && abs(poses_database[i].cam_stamp.toSec() - previous_stamp) < 0.01)
            {
                previouse_idx = i;
            }
            if (current_idx < 0 && abs(poses_database[i].cam_stamp.toSec() - current_stamp) < 0.01)
            {
                current_idx = i;
            }
        }
        if (previouse_idx > 0 && current_idx > 0)
        {
            printf("The loop index link %d -> %d\n", previouse_idx, current_idx);
            poses_database[previouse_idx].linked_pose_index.push_back(current_idx);
            poses_database[current_idx].linked_pose_index.push_back(previouse_idx);
            linked_pose_stamps[linked_i].first = -1.0;
            linked_pose_stamps[linked_i].second = -1.0;
        }
        else
        {
            printf("cannot find the associated poses!!!");
        }
    }
}


int lst_path_size;
void SurfelMap::path_input(const nav_msgs::PathConstPtr &loop_path_input)
{   
    cout<<"loop_path_input size: ="<<loop_path_input->poses.size()<<endl;
    cout<<"poses_database size: ="<<poses_database.size()<<endl;

/*    for(int i = loop_path_input->poses.size() - 2; i >= 0; i--)
    {
        if( fabs(loop_path_input->poses.back().header.stamp.toSec() - loop_path_input->poses[i].header.stamp.toSec()) <= 0.01 )
            ROS_BREAK();
    }

    for(int i = loop_path_input->poses.size() - 1; i >= 1; i--)
    {   
        for(int j =  i - 1; j >= 0; j--)
            if( loop_path_input->poses[i].header.stamp.toSec() <= loop_path_input->poses[j].header.stamp.toSec())
            {
                cout<<"2"<<endl;
                ROS_BREAK();
            }
    }
*/
    if(lst_path_size == loop_path_input->poses.size())
        return;
    
    lst_path_size = loop_path_input->poses.size();

    if(is_first_path || (!extrinsic_matrix_initialized))
    {
        is_first_path = false;
        pre_path_delete_time = loop_path_input->poses.back().header.stamp.toSec();
        return;
    }


    //printf("\nbegin new frame process!!!\n");

    // Eigen::Matrix4d imu2cam = Eigen::Matrix4d::Identity();
    // imu2cam(0,0) = Ric00;
    // imu2cam(0,1) = Ric01;
    // imu2cam(0,2) = Ric02;
    // imu2cam(1,0) = Ric10;
    // imu2cam(1,1) = Ric11;
    // imu2cam(1,2) = Ric12;
    // imu2cam(2,0) = Ric20;
    // imu2cam(2,1) = Ric21;
    // imu2cam(2,2) = Ric22;
    // imu2cam(0,3) = Tic0;
    // imu2cam(1,3) = Tic1;
    // imu2cam(2,3) = Tic2;
    //std::cout << "imu2cam" << std::endl << imu2cam << std::endl;

    nav_msgs::Path camera_path;
    geometry_msgs::PoseStamped cam_posestamped;
    for(int i = 0; i < loop_path_input->poses.size(); i++)
    {
        geometry_msgs::PoseStamped imu_posestamped = loop_path_input->poses[i];
        if(imu_posestamped.header.stamp.toSec() < pre_path_delete_time)
            continue;
        cam_posestamped = imu_posestamped;
        Eigen::Matrix4d imu_t, cam_t;
        pose_ros2eigen(imu_posestamped.pose, imu_t);
        cam_t = imu_t * extrinsic_matrix;
        pose_eigen2ros(cam_t, cam_posestamped.pose);
        camera_path.poses.push_back(cam_posestamped);
    }

    cam_posestamped.header.frame_id = "world";
    cam_pose_publish.publish(cam_posestamped);

    bool have_new_pose = false;
    /*geometry_msgs::Pose input_pose;
    if(camera_path.poses.size() > poses_database.size())
    {
        input_pose = camera_path.poses.back().pose;
        have_new_pose = true;
    }*/

    if(camera_path.poses.size() > poses_database.size())
    {
        have_new_pose = true;
    }

    // first update the poses
    bool loop_changed = false;
    for(int i = 0; i < poses_database.size() && i < camera_path.poses.size(); i++)
    {
        poses_database[i].loop_pose = camera_path.poses[i].pose;
        int this_world_id, this_set_id;
        get_world_set_id(camera_path.poses[i].header.frame_id, this_world_id, this_set_id);
        poses_database[i].set_id = this_set_id;
        if (poses_database[i].world_id != this_world_id)
        {
            std::cout << "previouse world ID: " << poses_database[i].world_id << std::endl;
            std::cout << "this world ID: " << this_world_id << std::endl;
            std::cout << "this stamp: " << camera_path.poses[i].header.stamp.toNSec() << std::endl;
            std::cout << camera_path.poses[i].header.frame_id << std::endl;
        }
        assert(poses_database[i].world_id == this_world_id);

        if( poses_database[i].loop_pose.position.x != poses_database[i].cam_pose.position.x
            || poses_database[i].loop_pose.position.y != poses_database[i].cam_pose.position.y
            || poses_database[i].loop_pose.position.z != poses_database[i].cam_pose.position.z)
        {
            loop_changed = true;
        }
    }

    //printf("warp the surfels according to the loop!\n");
    std::chrono::time_point<std::chrono::system_clock> start_time, end_time;
    start_time = std::chrono::system_clock::now();
    if(loop_changed)
    {
        warp_surfels();
    }
    end_time = std::chrono::system_clock::now();
    std::chrono::duration<double> used_time = end_time - start_time;
    double all_time = used_time.count() * 1000.0;

    if(have_new_pose)
    {
        // add new pose
        for(int i = 0; i < camera_path.poses.size(); i ++)
        {   
            if(poses_database.size() > 0)
            {
                if(camera_path.poses[i].header.stamp.toSec() > poses_database.back().cam_stamp.toSec())
                {
                    PoseElement this_pose_element;
                    int this_pose_index = poses_database.size();
                    this_pose_element.cam_pose  = camera_path.poses[i].pose;
                    this_pose_element.loop_pose = camera_path.poses[i].pose;
                    this_pose_element.cam_stamp = camera_path.poses[i].header.stamp;
                    int this_world_id, this_set_id;
                    get_world_set_id(camera_path.poses[i].header.frame_id, this_world_id, this_set_id);
                    this_pose_element.world_id = this_world_id;
                    this_pose_element.set_id = this_set_id;

                    int relative_index = poses_database.size() - 1;
                    if(poses_database[relative_index].world_id == this_pose_element.world_id)
                    {
                        this_pose_element.linked_pose_index.push_back(relative_index);
                        poses_database[relative_index].linked_pose_index.push_back(this_pose_index);
                    }

                    poses_database.push_back(this_pose_element);
                    local_surfels_indexs.insert(this_pose_index);

                    pose_reference_buffer.push_back(std::make_pair(camera_path.poses[i].header.stamp, this_pose_index));
                }
            }
            else
            {
                PoseElement this_pose_element;
                int this_pose_index = poses_database.size();
                this_pose_element.cam_pose  = camera_path.poses[i].pose;
                this_pose_element.loop_pose = camera_path.poses[i].pose;
                this_pose_element.cam_stamp = camera_path.poses[i].header.stamp;
                int this_world_id, this_set_id;
                get_world_set_id(camera_path.poses[i].header.frame_id, this_world_id, this_set_id);
                this_pose_element.world_id = this_world_id;
                this_pose_element.set_id = this_set_id;

                poses_database.push_back(this_pose_element);
                local_surfels_indexs.insert(this_pose_index);

                pose_reference_buffer.push_back(std::make_pair(camera_path.poses[i].header.stamp, this_pose_index));
            }
        }
        synchronize_msgs();
    }

    // push the msg into the buffer for fusion

}

void SurfelMap::pose_ros2eigen(geometry_msgs::Pose &pose, Eigen::Matrix4d &T)
{
    T = Eigen::Matrix4d::Identity();
    Eigen::Quaterniond rotation_q;
    rotation_q.w() = pose.orientation.w;
    rotation_q.x() = pose.orientation.x;
    rotation_q.y() = pose.orientation.y;
    rotation_q.z() = pose.orientation.z;
    T.block<3,3>(0,0) = rotation_q.toRotationMatrix();
    T(0,3) = pose.position.x;
    T(1,3) = pose.position.y;
    T(2,3) = pose.position.z;
}

void SurfelMap::pose_eigen2ros(Eigen::Matrix4d &T, geometry_msgs::Pose &pose)
{
    Eigen::Quaterniond rotation_q(T.block<3,3>(0,0));
    pose.orientation.w = rotation_q.w();
    pose.orientation.x = rotation_q.x();
    pose.orientation.y = rotation_q.y();
    pose.orientation.z = rotation_q.z();
    pose.position.x = T(0,3);
    pose.position.y = T(1,3);
    pose.position.z = T(2,3);
}

void SurfelMap::warp_inactive_surfels_cpu_kernel(int thread_i, int thread_num)
{
    std::chrono::time_point<std::chrono::system_clock> start_time, end_time;
    start_time = std::chrono::system_clock::now();
    int step = poses_database.size() / thread_num;
    int begin_index = step * thread_i;
    int end_index = begin_index + step;
    if (thread_i == thread_num - 1)
        end_index = poses_database.size();

    for(int i = begin_index; i < end_index; i ++)
    {
        if( poses_database[i].cam_pose.position.x == poses_database[i].loop_pose.position.x &&
            poses_database[i].cam_pose.position.y == poses_database[i].loop_pose.position.y &&
            poses_database[i].cam_pose.position.z == poses_database[i].loop_pose.position.z
            )
            continue;
        if(poses_database[i].attached_surfels.size() == 0)
        {
            poses_database[i].cam_pose = poses_database[i].loop_pose;
            continue;
        }

        PointCloud::Ptr warped_new_pointcloud(new PointCloud);

        Eigen::Matrix4d pre_pose, after_pose;
        Eigen::Matrix4f warp_matrix;
        pose_ros2eigen(poses_database[i].cam_pose, pre_pose);
        pose_ros2eigen(poses_database[i].loop_pose, after_pose);
        warp_matrix = (after_pose * pre_pose.inverse()).cast<float>();
        Eigen::MatrixXf point_positions(4, poses_database[i].attached_surfels.size());
        Eigen::MatrixXf point_norms(3, poses_database[i].attached_surfels.size());
        for(int surfel_i = 0; surfel_i < poses_database[i].attached_surfels.size(); surfel_i++)
        {
            point_positions(0,surfel_i) = poses_database[i].attached_surfels[surfel_i].px;
            point_positions(1,surfel_i) = poses_database[i].attached_surfels[surfel_i].py;
            point_positions(2,surfel_i) = poses_database[i].attached_surfels[surfel_i].pz;
            point_positions(3,surfel_i) = 1.0;
            point_norms(0,surfel_i) = poses_database[i].attached_surfels[surfel_i].nx;
            point_norms(1,surfel_i) = poses_database[i].attached_surfels[surfel_i].ny;
            point_norms(2,surfel_i) = poses_database[i].attached_surfels[surfel_i].nz;
        }
        point_positions = warp_matrix * point_positions;
        point_norms = warp_matrix.block<3,3>(0,0) * point_norms;
        for(int surfel_i = 0; surfel_i < poses_database[i].attached_surfels.size(); surfel_i++)
        {
            poses_database[i].attached_surfels[surfel_i].px = point_positions(0,surfel_i);
            poses_database[i].attached_surfels[surfel_i].py = point_positions(1,surfel_i);
            poses_database[i].attached_surfels[surfel_i].pz = point_positions(2,surfel_i);
            poses_database[i].attached_surfels[surfel_i].nx = point_norms(0,surfel_i);
            poses_database[i].attached_surfels[surfel_i].ny = point_norms(1,surfel_i);
            poses_database[i].attached_surfels[surfel_i].nz = point_norms(2,surfel_i);

            PointType new_point;
            new_point.x = poses_database[i].attached_surfels[surfel_i].px;
            new_point.y = poses_database[i].attached_surfels[surfel_i].py;
            new_point.z = poses_database[i].attached_surfels[surfel_i].pz;
            new_point.intensity = poses_database[i].attached_surfels[surfel_i].color;
            warped_new_pointcloud->push_back(new_point);
        }
        poses_database[i].cam_pose = poses_database[i].loop_pose;
        std::copy(&warped_new_pointcloud->front(), &warped_new_pointcloud->back(), &inactive_pointcloud->at(poses_database[i].points_begin_index));
    }
    end_time = std::chrono::system_clock::now();
    std::chrono::duration<double> used_time = end_time - start_time;
    double all_time = used_time.count() * 1000.0;
//    printf("warp kernel %d, cost %f ms.\n", thread_i, all_time);
}

void SurfelMap::warp_active_surfels_cpu_kernel(int thread_i, int thread_num, Eigen::Matrix4f transform_m)
{
    std::chrono::time_point<std::chrono::system_clock> start_time, end_time;
    start_time = std::chrono::system_clock::now();
    int step = local_surfels.size() / thread_num;
    int begin_index = step * thread_i;
    int end_index = begin_index + step;
    if (thread_i == thread_num - 1)
        end_index = local_surfels.size();
    int surfel_num = end_index - begin_index;

    Eigen::MatrixXf point_positions(4, surfel_num);
    Eigen::MatrixXf point_norms(3, surfel_num);
    for(int i = 0; i < surfel_num; i++)
    {
        point_positions(0, i) = local_surfels[i + begin_index].px;
        point_positions(1, i) = local_surfels[i + begin_index].py;
        point_positions(2, i) = local_surfels[i + begin_index].pz;
        point_positions(3, i) = 1.0;
        point_norms(0, i) = local_surfels[i + begin_index].nx;
        point_norms(1, i) = local_surfels[i + begin_index].ny;
        point_norms(2, i) = local_surfels[i + begin_index].nz;
    }
    point_positions = transform_m * point_positions;
    point_norms = transform_m.block<3,3>(0,0) * point_norms;
    for(int i = 0; i < surfel_num; i++)
    {
        local_surfels[i + begin_index].px = point_positions(0, i);
        local_surfels[i + begin_index].py = point_positions(1, i);
        local_surfels[i + begin_index].pz = point_positions(2, i);
        local_surfels[i + begin_index].nx = point_norms(0, i);
        local_surfels[i + begin_index].ny = point_norms(1, i);
        local_surfels[i + begin_index].nz = point_norms(2, i);
    }

    end_time = std::chrono::system_clock::now();
    std::chrono::duration<double> used_time = end_time - start_time;
    double all_time = used_time.count() * 1000.0;
    // printf("warp kernel %d, cost %f ms.\n", thread_i, all_time);
}

void SurfelMap::warp_surfels()
{
    warp_thread_pool.clear();
    warp_thread_num = 10;
    // warp inactive surfels
    for(int i = 0; i < warp_thread_num; i++)
    {
        std::thread this_thread(&SurfelMap::warp_inactive_surfels_cpu_kernel, this, i, warp_thread_num);
        warp_thread_pool.push_back(std::move(this_thread));
    }

    // for(int i = 0; i < warp_thread_pool.size(); i++)
    //     if(warp_thread_pool[i].joinable())
    //         warp_thread_pool[i].join();
    // warp_thread_pool.clear();

    // warp active surfels
    int local_index = *local_surfels_indexs.begin();
    Eigen::Matrix4d pre_pose, loop_pose;
    Eigen::Matrix4f warp_pose;
    pose_ros2eigen(poses_database[local_index].cam_pose, pre_pose);
    pose_ros2eigen(poses_database[local_index].loop_pose, loop_pose);
    warp_pose = (loop_pose * pre_pose.inverse()).cast<float>();
    
    for(int i = 0; i < warp_thread_num; i++)
    {
        std::thread this_thread(&SurfelMap::warp_active_surfels_cpu_kernel, this, i, warp_thread_num, warp_pose);
        warp_thread_pool.push_back(std::move(this_thread));
    }

    for(int i = 0; i < warp_thread_pool.size(); i++)
        if(warp_thread_pool[i].joinable())
            warp_thread_pool[i].join();
}

void SurfelMap::calculate_memory_usage()
{
    double usgae_KB = 0;
    usgae_KB += local_surfels.size() * sizeof(SurfelElement)  / 1024.0;
    usgae_KB += poses_database.size() * sizeof(PoseElement) / 1024.0;
    usgae_KB += local_surfels_indexs.size() * sizeof(int) / 1024.0;
    // usgae_KB += inactive_pointcloud->size() * sizeof(PointType) / 1024.0;
    usgae_KB += inactive_pointcloud->size() * sizeof(SurfelElement)  / 1024.0;
    //printf("the process comsumes %f KB\n", usgae_KB);
}
void SurfelMap::publish_pose_graph(ros::Time pub_stamp, int reference_index)
{
    nav_msgs::Path loop_path;
    loop_path.header.stamp = pub_stamp;
    loop_path.header.frame_id = "world";

    visualization_msgs::Marker loop_marker;
    loop_marker.header.frame_id = "world";
    loop_marker.header.stamp = pub_stamp;
    loop_marker.ns = "namespace";
    loop_marker.id = 0;
    loop_marker.type = visualization_msgs::Marker::LINE_LIST;
    loop_marker.action = visualization_msgs::Marker::ADD;
    loop_marker.scale.x = 0.01;
    loop_marker.scale.y = 0.01;
    loop_marker.scale.z = 0.01;
    loop_marker.color.a = 1.0; // Don't forget to set the alpha!
    loop_marker.color.r = 1.0;
    loop_marker.color.g = 0.0;
    loop_marker.color.b = 0.0;
    for(int i = 0; i < poses_database.size(); i++)
    {
        geometry_msgs::PoseStamped loop_pose;
        loop_pose.header.stamp = poses_database[i].cam_stamp;
        loop_pose.pose = poses_database[i].cam_pose;

        loop_path.poses.push_back(loop_pose);

        for(int j = 0; j < poses_database[i].linked_pose_index.size(); j++)
        {
            if(     poses_database[i].linked_pose_index[j] != i-1 
                &&  poses_database[i].linked_pose_index[j] != i+1
                &&  poses_database[i].linked_pose_index[j] > i
                )
            {
                geometry_msgs::Point one_point, another_point;
                one_point.x = poses_database[i].loop_pose.position.x;
                one_point.y = poses_database[i].loop_pose.position.y;
                one_point.z = poses_database[i].loop_pose.position.z;
                another_point.x = poses_database[poses_database[i].linked_pose_index[j]].loop_pose.position.x;
                another_point.y = poses_database[poses_database[i].linked_pose_index[j]].loop_pose.position.y;
                another_point.z = poses_database[poses_database[i].linked_pose_index[j]].loop_pose.position.z;
                loop_marker.points.push_back(one_point);
                loop_marker.points.push_back(another_point);
            }
        }
    }

    loop_path_publish.publish(loop_path);
    loop_marker_publish.publish(loop_marker);

    // publish driftfree poses
    visualization_msgs::Marker driftfree_marker;
    driftfree_marker.header.frame_id = "world";
    driftfree_marker.header.stamp = pub_stamp;
    driftfree_marker.ns = "namespace";
    driftfree_marker.id = 0;
    driftfree_marker.type = visualization_msgs::Marker::SPHERE_LIST;
    driftfree_marker.action = visualization_msgs::Marker::ADD;
    driftfree_marker.scale.x = 0.2;
    driftfree_marker.scale.y = 0.2;
    driftfree_marker.scale.z = 0.2;
    driftfree_marker.color.a = 1.0; // Don't forget to set the alpha!
    driftfree_marker.color.r = 1.0;
    driftfree_marker.color.g = 0.0;
    driftfree_marker.color.b = 0.0;
    vector<int> driftfree_indexs;
    get_driftfree_poses(reference_index, driftfree_indexs, drift_free_poses);
    for(int i = 0; i < driftfree_indexs.size(); i++)
    {
        geometry_msgs::Point one_point;
        one_point.x = poses_database[driftfree_indexs[i]].cam_pose.position.x;
        one_point.y = poses_database[driftfree_indexs[i]].cam_pose.position.y;
        one_point.z = poses_database[driftfree_indexs[i]].cam_pose.position.z;
        driftfree_marker.points.push_back(one_point);
    }
    driftfree_path_publish.publish(driftfree_marker);
}


void SurfelMap::fuse_map(cv::Mat image, cv::Mat depth, Eigen::Matrix4f pose_input, int reference_index)
{
    //printf("fuse surfels with reference index %d and %d surfels!\n", reference_index, local_surfels.size());    
    Timer fuse_timer("fusing");

    vector<SurfelElement> new_surfels;
    fusion_functions.fuse_initialize_map(
        reference_index,
        image,
        depth,
        pose_input,
        local_surfels,
        new_surfels
    );
    fuse_timer.middle("fuse_initialize_map");

    // get the deleted surfel index
    vector<int> deleted_index;
    for(int i = 0; i < local_surfels.size(); i++)
    {
        if(local_surfels[i].update_times == 0)
            deleted_index.push_back(i);
    }
    fuse_timer.middle("delete index");

    // add new initialized surfels
    int add_surfel_num = 0;
    for(int i = 0; i < new_surfels.size(); i++)
    {
        if(new_surfels[i].update_times != 0)
        {
            SurfelElement this_surfel = new_surfels[i];
            if(deleted_index.size() > 0)
            {
                local_surfels[deleted_index.back()] = this_surfel;
                deleted_index.pop_back();
            }
            else
                local_surfels.push_back(this_surfel);
            add_surfel_num += 1;
        }
    }
    // remove deleted surfels
    while(deleted_index.size() > 0)
    {
        local_surfels[deleted_index.back()] = local_surfels.back();
        deleted_index.pop_back();
        local_surfels.pop_back();
    }
    fuse_timer.middle("cpu part");
    //printf("add %d surfels, we now have %d local surfels.\n", add_surfel_num, local_surfels.size());
    fuse_timer.end();
}

void SurfelMap::publish_raw_pointcloud(cv::Mat &depth, cv::Mat &reference, geometry_msgs::Pose &pose)
{
    Eigen::Matrix3f rotation_R;
    Eigen::Vector3f translation_T;
    Eigen::Quaternionf rotation_q;
    rotation_q.w() = pose.orientation.w;
    rotation_q.x() = pose.orientation.x;
    rotation_q.y() = pose.orientation.y;
    rotation_q.z() = pose.orientation.z;
    rotation_R = rotation_q.toRotationMatrix();
    translation_T(0) = pose.position.x;
    translation_T(1) = pose.position.y;
    translation_T(2) = pose.position.z;

    PointCloud::Ptr pointcloud(new PointCloud);
    for(int i = 0; i < cam_width; i++)
    for(int j = 0; j < cam_height; j++)
    {
        float depth_value = depth.at<float>(j,i);
        Eigen::Vector3f cam_point;
        cam_point(0) = (i - cam_cx) * depth_value / cam_fx;
        cam_point(1) = (j - cam_cy) * depth_value / cam_fy;
        cam_point(2) = depth_value;
        Eigen::Vector3f world_point;
        world_point = rotation_R * cam_point + translation_T;

        PointType p;
        p.x = world_point(0);
        p.y = world_point(1);
        p.z = world_point(2);
        p.intensity = reference.at<uchar>(j,i);
        pointcloud->push_back(p);
    }
    pointcloud->header.frame_id = "world";
    raw_pointcloud_publish.publish(pointcloud);
    printf("publish raw point cloud with %d points.\n", pointcloud->size());
}

void SurfelMap::save_cloud(string save_path_name)
{
    printf("saving pointcloud ...\n");
    PointCloud::Ptr pointcloud(new PointCloud);
    for(int surfel_it = 0; surfel_it < local_surfels.size(); surfel_it++)
    {
        if(local_surfels[surfel_it].update_times < 5)
            continue;
        PointType p;
        p.x = local_surfels[surfel_it].px;
        p.y = local_surfels[surfel_it].py;
        p.z = local_surfels[surfel_it].pz;
        p.intensity = local_surfels[surfel_it].color;
        pointcloud->push_back(p);
    }
    
    (*pointcloud) += (*inactive_pointcloud);
    
    // pcl::io::savePLYFile(save_path_name.c_str(), *pointcloud);
    pcl::io::savePCDFile(save_path_name.c_str(), *pointcloud);
    printf("saving pointcloud done!\n");
}

void SurfelMap::push_a_surfel(vector<float> &vertexs, SurfelElement &this_surfel)
{
    int surfel_color = this_surfel.color;
    Eigen::Vector3f surfel_position;
    surfel_position(0) = this_surfel.px;
    surfel_position(1) = this_surfel.py;
    surfel_position(2) = this_surfel.pz;
    Eigen::Vector3f surfel_norm;
    surfel_norm(0) = this_surfel.nx;
    surfel_norm(1) = this_surfel.ny;
    surfel_norm(2) = this_surfel.nz;
    Eigen::Vector3f x_dir;
    x_dir(0) = -1 * this_surfel.ny;
    x_dir(1) = this_surfel.nx;
    x_dir(2) = 0;
    x_dir.normalize();
    Eigen::Vector3f y_dir;
    y_dir = surfel_norm.cross(x_dir);
    float radius = this_surfel.size;
    float h_r = radius * 0.5;
    float t_r = radius * 0.86603;
    Eigen::Vector3f point1, point2, point3, point4, point5, point6;
    point1 = surfel_position - x_dir * h_r - y_dir * t_r;
    point2 = surfel_position + x_dir * h_r - y_dir * t_r;
    point3 = surfel_position - x_dir * radius;
    point4 = surfel_position + x_dir * radius;
    point5 = surfel_position - x_dir * h_r + y_dir * t_r;
    point6 = surfel_position + x_dir * h_r + y_dir * t_r;
    vertexs.push_back(point1(0));vertexs.push_back(point1(1));vertexs.push_back(point1(2));
    vertexs.push_back(surfel_color);vertexs.push_back(surfel_color);vertexs.push_back(surfel_color);
    vertexs.push_back(point2(0));vertexs.push_back(point2(1));vertexs.push_back(point2(2));
    vertexs.push_back(surfel_color);vertexs.push_back(surfel_color);vertexs.push_back(surfel_color);
    vertexs.push_back(point3(0));vertexs.push_back(point3(1));vertexs.push_back(point3(2));
    vertexs.push_back(surfel_color);vertexs.push_back(surfel_color);vertexs.push_back(surfel_color);
    vertexs.push_back(point4(0));vertexs.push_back(point4(1));vertexs.push_back(point4(2));
    vertexs.push_back(surfel_color);vertexs.push_back(surfel_color);vertexs.push_back(surfel_color);
    vertexs.push_back(point5(0));vertexs.push_back(point5(1));vertexs.push_back(point5(2));
    vertexs.push_back(surfel_color);vertexs.push_back(surfel_color);vertexs.push_back(surfel_color);
    vertexs.push_back(point6(0));vertexs.push_back(point6(1));vertexs.push_back(point6(2));
    vertexs.push_back(surfel_color);vertexs.push_back(surfel_color);vertexs.push_back(surfel_color);
}


void SurfelMap::save_mesh(string save_path_name)
{
    std::ofstream stream(save_path_name.c_str());
    if (!stream)
        return;
    std::vector<float> vertexs;
    for(int i = 0; i < poses_database.size(); i++)
    {
        for(int j = 0; j < poses_database[i].attached_surfels.size(); j++)
        {
            SurfelElement this_surfel = poses_database[i].attached_surfels[j];
            push_a_surfel(vertexs, this_surfel);
        }
    }

    for(int i = 0; i < local_surfels.size(); i++)
    {
        if(local_surfels[i].update_times < 5)
            continue;
        SurfelElement this_surfel = local_surfels[i];
        push_a_surfel(vertexs, this_surfel);
    }
    
    size_t numPoints = vertexs.size()/6;
    size_t numSurfels = numPoints/6;
    stream << "ply" << std::endl;
    stream << "format ascii 1.0" << std::endl;
    stream << "element vertex " << numPoints << std::endl;
    stream << "property float x" << std::endl;
    stream << "property float y" << std::endl;
    stream << "property float z" << std::endl;
    stream << "property uchar red" << std::endl;
    stream << "property uchar green" << std::endl;
    stream << "property uchar blue" << std::endl;
    stream << "element face " << numSurfels * 4 <<  std::endl;
    stream << "property list uchar int vertex_index" << std::endl;
    stream << "end_header" << std::endl;

    for(int i = 0; i < numPoints; i++)
    {
        for(int j = 0; j < 6; j++)
        {
            stream << vertexs[i*6+j] << " ";
        }
        stream << std::endl;
    }
    for(int i = 0; i < numSurfels; i++)
    {
        int p1, p2, p3, p4, p5, p6;
        p1 = i * 6 + 0;
        p2 = i * 6 + 1;
        p3 = i * 6 + 2;
        p4 = i * 6 + 3;
        p5 = i * 6 + 4;
        p6 = i * 6 + 5;
        stream << "3 " << p1 << " " << p2 << " " << p3 << std::endl;
        stream << "3 " << p2 << " " << p4 << " " << p3 << std::endl;
        stream << "3 " << p3 << " " << p4 << " " << p5 << std::endl;
        stream << "3 " << p5 << " " << p4 << " " << p6 << std::endl;
    }
    stream.close();
}

void SurfelMap::publish_this_set_pointcloud(ros::Time pub_stamp, int reference_index)
{
    std::chrono::time_point<std::chrono::system_clock> start_time, end_time;
    std::chrono::duration<double> total_time;
    start_time = std::chrono::system_clock::now();

    PointCloud::Ptr pointcloud(new PointCloud);
    // pointcloud->reserve(local_surfels.size() + inactive_pointcloud->size());
    for (int surfel_it = 0; surfel_it < local_surfels.size(); surfel_it++)
    {
        if (local_surfels[surfel_it].update_times == 0)
            continue;
        PointType p;
        p.x = local_surfels[surfel_it].px;
        p.y = local_surfels[surfel_it].py;
        p.z = local_surfels[surfel_it].pz;
        p.intensity = local_surfels[surfel_it].color;
        pointcloud->push_back(p);
    }

    //METHOD 1, NAIVE ADD THE POINTS
    std::vector<int> neighbor_indexs;
    int this_set_id = poses_database[reference_index].set_id;
    for (int i = 0; i < poses_database.size(); i++)
    {
        if(poses_database[i].set_id == this_set_id)
            neighbor_indexs.push_back(i);
    }
    for (int i = 0; i < neighbor_indexs.size(); i++)
    {
        int this_pose = neighbor_indexs[i];
        if (local_surfels_indexs.find(this_pose) != local_surfels_indexs.end())
            continue;
        int pointcloud_num = poses_database[this_pose].attached_surfels.size();
        int pointcloud_begin = poses_database[this_pose].points_begin_index;
        if (pointcloud_num <= 0)
            continue;
        pointcloud->insert(
            pointcloud->end(),
            inactive_pointcloud->begin() + pointcloud_begin,
            inactive_pointcloud->begin() + pointcloud_begin + pointcloud_num);
    }
    //NETHOD 1 ENDS

    end_time = std::chrono::system_clock::now();
    total_time = end_time - start_time;
    //printf("construct point cloud cost %f ms.\n", total_time.count()*1000.0);
    start_time = std::chrono::system_clock::now();

    pointcloud->header.frame_id = "world";
    pcl_conversions::toPCL(pub_stamp, pointcloud->header.stamp);
    set_pointcloud_publish.publish(pointcloud);
    //    printf("publish point cloud with %d points, in active %d points.\n", pointcloud->size(), inactive_pointcloud->size());

    end_time = std::chrono::system_clock::now();
    total_time = end_time - start_time;
    //printf("publish point cloud cost %f ms.\n", total_time.count()*1000.0);
}

void SurfelMap::publish_neighbor_pointcloud(ros::Time pub_stamp, int reference_index)
{
    std::chrono::time_point<std::chrono::system_clock> start_time, end_time;
    std::chrono::duration<double> total_time;
    start_time = std::chrono::system_clock::now();

    PointCloud::Ptr pointcloud(new PointCloud);
    pointcloud->reserve(local_surfels.size() + inactive_pointcloud->size());
    for(int surfel_it = 0; surfel_it < local_surfels.size(); surfel_it++)
    {
        if(local_surfels[surfel_it].update_times == 0)
            continue;
        PointType p;
        p.x = local_surfels[surfel_it].px;
        p.y = local_surfels[surfel_it].py;
        p.z = local_surfels[surfel_it].pz;
        p.intensity = local_surfels[surfel_it].color;
        pointcloud->push_back(p);
    }

    // add other pointcloud
    
    //METHOD 1, NAIVE ADD THE POINTS
    std::vector<int> neighbor_indexs;
    get_driftfree_poses(reference_index, neighbor_indexs, 2*drift_free_poses);
    for(int i = 0; i < neighbor_indexs.size(); i++)
    {
        int this_pose = neighbor_indexs[i];
        if(local_surfels_indexs.find(this_pose) != local_surfels_indexs.end())
            continue;
        int pointcloud_num = poses_database[this_pose].attached_surfels.size();
        int pointcloud_begin = poses_database[this_pose].points_begin_index;
        if(pointcloud_num <= 0)
            continue;
        pointcloud->insert(
            pointcloud->end(),
            inactive_pointcloud->begin()+pointcloud_begin,
            inactive_pointcloud->begin()+pointcloud_begin+pointcloud_num);
    }
    //NETHOD 1 ENDS

    // //METHOD 2, FIND THE SUCCESSIVELY MEMORY AND ADD
    // std::vector<int> neighbor_indexs;
    // get_driftfree_poses(reference_index, neighbor_indexs, 2*drift_free_poses);
    // std::vector<int> points_begin_end;
    // for(int i = 0; i < neighbor_indexs.size(); i++)
    // {
    //     int this_pose = neighbor_indexs[i];
    //     if(local_surfels_indexs.find(this_pose) != local_surfels_indexs.end())
    //         continue;
    //     int pointcloud_num = poses_database[this_pose].attached_surfels.size();
    //     int pointcloud_begin = poses_database[this_pose].points_begin_index;
    //     if(pointcloud_num <= 0)
    //         continue;
    //     points_begin_end.push_back(pointcloud_begin);
    //     points_begin_end.push_back(pointcloud_begin+pointcloud_num);
    // }
    // if(points_begin_end.size() > 0)
    // {
    //     std::sort(points_begin_end.begin(), points_begin_end.end());
    //     int points_add_begin = points_begin_end.front();
    //     bool need_to_add = false;
    //     for(int i = 0; i < points_begin_end.size() / 2; i++)
    //     {
    //         if(need_to_add)
    //             points_add_begin = points_begin_end[2*i];
    //         need_to_add = false;
    //         int this_end = points_begin_end[2*i+1];
    //         if(i == points_begin_end.size() / 2 - 1)
    //             need_to_add = true;
    //         else
    //         {
    //             int next_begin = points_begin_end[2*i + 2];
    //             if(next_begin != this_end + 1)
    //                 need_to_add = true;
    //         }
    //         if(need_to_add)
    //         {
    //             pointcloud->insert(
    //                 pointcloud->end(),
    //                 inactive_pointcloud->begin()+points_add_begin,
    //                 inactive_pointcloud->begin()+this_end);
    //         }
    //     }
    // }
    // //METHOD 2 ENDS

    end_time = std::chrono::system_clock::now();
    total_time = end_time - start_time;
    //printf("construct point cloud cost %f ms.\n", total_time.count()*1000.0);
    start_time = std::chrono::system_clock::now();

    pointcloud->header.frame_id = "world";
    pcl_conversions::toPCL(pub_stamp, pointcloud->header.stamp);
    pointcloud_publish.publish(pointcloud);
//    printf("publish point cloud with %d points, in active %d points.\n", pointcloud->size(), inactive_pointcloud->size());

    end_time = std::chrono::system_clock::now();
    total_time = end_time - start_time;
    //printf("publish point cloud cost %f ms.\n", total_time.count()*1000.0);
}


void SurfelMap::publish_all_pointcloud(ros::Time pub_stamp)
{
    std::chrono::time_point<std::chrono::system_clock> start_time, end_time;
    std::chrono::duration<double> total_time;
    start_time = std::chrono::system_clock::now();

    PointCloud::Ptr pointcloud(new PointCloud);
    pointcloud->reserve(local_surfels.size() + inactive_pointcloud->size());
    for(int surfel_it = 0; surfel_it < local_surfels.size(); surfel_it++)
    {
        if(local_surfels[surfel_it].update_times < 5)
            continue;
        PointType p;
        p.x = local_surfels[surfel_it].px;
        p.y = local_surfels[surfel_it].py;
        p.z = local_surfels[surfel_it].pz;
        p.intensity = local_surfels[surfel_it].color;
        pointcloud->push_back(p);
    }

    (*pointcloud) += (*inactive_pointcloud);

    end_time = std::chrono::system_clock::now();
    total_time = end_time - start_time;
    // printf("construct point cloud cost %f ms.\n", total_time.count()*1000.0);
    start_time = std::chrono::system_clock::now();

/*    pointcloud->header.frame_id = "world";
    pcl_conversions::toPCL(pub_stamp, pointcloud->header.stamp);*/

    PointCloud::Ptr pointcloud_noceil(new PointCloud);
    for(int i = 0; i < pointcloud->points.size(); i++)
    {   
        /*if( pointcloud->points[i].z > 2.2 )
            continue;*/
        pointcloud_noceil->points.push_back(pointcloud->points[i]);
    }

    pointcloud_noceil->header.frame_id = "world";
    pcl_conversions::toPCL(pub_stamp, pointcloud_noceil->header.stamp);
    pointcloud_publish.publish(pointcloud_noceil);

    //pointcloud_publish.publish(pointcloud);
    //printf("publish point cloud with %d points, inactive %d points.\n", pointcloud->size(), inactive_pointcloud->size());

    // end_time = std::chrono::system_clock::now();
    // total_time = end_time - start_time;
    // printf("publish point cloud cost %f ms.\n", total_time.count()*1000.0);
}

void SurfelMap::move_all_surfels()
{
    vector<int> poses_to_remove(local_surfels_indexs.begin(), local_surfels_indexs.end());
    std::chrono::time_point<std::chrono::system_clock> start_time, end_time;
    std::chrono::duration<double> move_pointcloud_time;

    if(poses_to_remove.size() > 0)
    {
        
        start_time = std::chrono::system_clock::now();
        int added_surfel_num = 0;
        float sum_update_times = 0.0;
        for(int pi = 0; pi < poses_to_remove.size(); pi++)
        {
            int inactive_index = poses_to_remove[pi];
            poses_database[inactive_index].points_begin_index = inactive_pointcloud->size();
            poses_database[inactive_index].points_pose_index = pointcloud_pose_index.size();
            pointcloud_pose_index.push_back(inactive_index);
            for(int i = 0; i < local_surfels.size(); i++)
            {
                if(local_surfels[i].update_times > 0 && local_surfels[i].last_update == inactive_index)
                {
                    poses_database[inactive_index].attached_surfels.push_back(local_surfels[i]);

                    PointType p;
                    p.x = local_surfels[i].px;
                    p.y = local_surfels[i].py;
                    p.z = local_surfels[i].pz;
                    p.intensity = local_surfels[i].color;
                    inactive_pointcloud->push_back(p);

                    added_surfel_num += 1;
                    sum_update_times += local_surfels[i].update_times;

                    // delete the surfel from the local point
                    local_surfels[i].update_times = 0;
                }
            }
            // printf("remove pose %d from local poses, get %d surfels.\n", inactive_index, poses_database[inactive_index].attached_surfels.size());
            local_surfels_indexs.erase(inactive_index);
        }
        sum_update_times = sum_update_times / added_surfel_num;
        end_time = std::chrono::system_clock::now();
        move_pointcloud_time = end_time - start_time;
        //printf("move surfels cost %f ms. the average update times is %f.\n", move_pointcloud_time.count()*1000.0, sum_update_times);
    }
}

void SurfelMap::move_add_surfels(int reference_index)
{
    // remove inactive surfels
    //printf("get inactive surfels for pose %d.\n", reference_index);
    // vector<int> drift_poses;
    vector<int> poses_to_add;
    vector<int> poses_to_remove;
    get_add_remove_poses(reference_index, poses_to_add, poses_to_remove);
    std::chrono::time_point<std::chrono::system_clock> start_time, end_time;
    std::chrono::duration<double> move_pointcloud_time;

    if(poses_to_remove.size() > 0)
    {
        
        start_time = std::chrono::system_clock::now();
        int added_surfel_num = 0;
        float sum_update_times = 0.0;
        for(int pi = 0; pi < poses_to_remove.size(); pi++)
        {
            int inactive_index = poses_to_remove[pi];
            poses_database[inactive_index].points_begin_index = inactive_pointcloud->size();
            poses_database[inactive_index].points_pose_index = pointcloud_pose_index.size();
            pointcloud_pose_index.push_back(inactive_index);
            for(int i = 0; i < local_surfels.size(); i++)
            {
                if(local_surfels[i].update_times > 0 && local_surfels[i].last_update == inactive_index)
                {
                    poses_database[inactive_index].attached_surfels.push_back(local_surfels[i]);

                    PointType p;
                    p.x = local_surfels[i].px;
                    p.y = local_surfels[i].py;
                    p.z = local_surfels[i].pz;
                    p.intensity = local_surfels[i].color;
                    inactive_pointcloud->push_back(p);

                    added_surfel_num += 1;
                    sum_update_times += local_surfels[i].update_times;

                    // delete the surfel from the local point
                    local_surfels[i].update_times = 0;
                }
            }
            //printf("remove pose %d from local poses, get %d surfels.\n", inactive_index, poses_database[inactive_index].attached_surfels.size());
            local_surfels_indexs.erase(inactive_index);
        }
        sum_update_times = sum_update_times / added_surfel_num;
        end_time = std::chrono::system_clock::now();
        move_pointcloud_time = end_time - start_time;
        //printf("move surfels cost %f ms. the average update times is %f.\n", move_pointcloud_time.count()*1000.0, sum_update_times);
    }
    if(poses_to_add.size() > 0)
    {
        // 1.0 add indexs
        local_surfels_indexs.insert(poses_to_add.begin(), poses_to_add.end());
        // 2.0 add surfels
        // 2.1 remove from inactive_pointcloud
        start_time = std::chrono::system_clock::now();
        std::vector<std::pair<int, int>> remove_info;//first, pointcloud start, pointcloud size, pointcloud pose index
        for(int add_i = 0; add_i < poses_to_add.size(); add_i++)
        {
            int add_index = poses_to_add[add_i];
            int pointcloud_pose_index = poses_database[add_index].points_pose_index;
            remove_info.push_back(std::make_pair(pointcloud_pose_index, add_index));
        }
        std::sort(
        remove_info.begin(),
        remove_info.end(),
        []( const std::pair<int, int >& first, const std::pair<int, int>& second)
        {
            return first.first < second.first;
        }
        );
        int remove_begin_index = remove_info[0].second;
        int remove_points_size = poses_database[remove_begin_index].attached_surfels.size();
        int remove_pose_size = 1;
        for(int remove_i = 1; remove_i <= remove_info.size(); remove_i++)
        {
            bool need_remove = false;
            if(remove_i == remove_info.size())
                need_remove = true;
            if(remove_i < remove_info.size())
            {
                if(remove_info[remove_i].first != (remove_info[remove_i-1].first + 1))
                    need_remove = true;
            }
            if(!need_remove)
            {
                int this_pose_index = remove_info[remove_i].second;
                remove_points_size += poses_database[this_pose_index].attached_surfels.size();
                remove_pose_size += 1;
                continue;
            }

            int remove_end_index = remove_info[remove_i - 1].second;
            //printf("remove from pose %d -> %d, has %d points\n", remove_begin_index, remove_end_index, remove_points_size);

            PointCloud::iterator begin_ptr;
            PointCloud::iterator end_ptr;
            begin_ptr = inactive_pointcloud->begin() + poses_database[remove_begin_index].points_begin_index;
            end_ptr = begin_ptr + remove_points_size;
            inactive_pointcloud->erase(begin_ptr, end_ptr);
            
            for(int pi = poses_database[remove_end_index].points_pose_index + 1; pi < pointcloud_pose_index.size(); pi++)
            {
                poses_database[pointcloud_pose_index[pi]].points_begin_index -= remove_points_size;
                poses_database[pointcloud_pose_index[pi]].points_pose_index -= remove_pose_size; 
            }
    
            pointcloud_pose_index.erase(
                pointcloud_pose_index.begin() + poses_database[remove_begin_index].points_pose_index,
                pointcloud_pose_index.begin() + poses_database[remove_end_index].points_pose_index + 1
            );


            if(remove_i < remove_info.size())
            {
                remove_begin_index = remove_info[remove_i].second;;
                remove_points_size = poses_database[remove_begin_index].attached_surfels.size();
                remove_pose_size = 1;
            }
        }

        // 2.3 add the surfels into local
        for(int pi = 0; pi < poses_to_add.size(); pi++)
        {
            int pose_index = poses_to_add[pi];
            local_surfels.insert(
                local_surfels.end(),
                poses_database[pose_index].attached_surfels.begin(),
                poses_database[pose_index].attached_surfels.end());
            poses_database[pose_index].attached_surfels.clear();
            poses_database[pose_index].points_begin_index = -1;
            poses_database[pose_index].points_pose_index = -1;
        }
        end_time = std::chrono::system_clock::now();
        move_pointcloud_time = end_time - start_time;
        //printf("add surfels cost %f ms.\n", move_pointcloud_time.count()*1000.0);
    }
}

void SurfelMap::get_add_remove_poses(int root_index, vector<int> &pose_to_add, vector<int> &pose_to_remove)
{
    vector<int> driftfree_poses;
    get_driftfree_poses(root_index, driftfree_poses, drift_free_poses);
    {
/*        printf("\ndriftfree poses: ");
        for(int i = 0; i < driftfree_poses.size(); i++)
        {
            printf("%d, ", driftfree_poses[i]);
        }*/
    }
    pose_to_add.clear();
    pose_to_remove.clear();
    // get to add
    for(int i = 0; i < driftfree_poses.size(); i++)
    {
        int temp_pose = driftfree_poses[i];
        if(local_surfels_indexs.find(temp_pose) == local_surfels_indexs.end())
            pose_to_add.push_back(temp_pose);
    }
    /*{
        printf("\nto add: ");
        for(int i = 0; i < pose_to_add.size(); i++)
        {
            printf("%d, ", pose_to_add[i]);
        }
    }*/
    // get to remove
    for(auto i = local_surfels_indexs.begin(); i != local_surfels_indexs.end(); i++)
    {
        int temp_pose = *i;
        if( std::find(driftfree_poses.begin(), driftfree_poses.end(), temp_pose) ==  driftfree_poses.end() )
        {
            pose_to_remove.push_back(temp_pose);
        }
    }
    {
        //printf("\nto remove: ");
        for(int i = 0; i < pose_to_remove.size(); i++)
        {
            //printf("%d, ", pose_to_remove[i]);
        }
        //printf("\n");
    }
}

void SurfelMap::get_driftfree_poses(int root_index, vector<int> &driftfree_poses, int driftfree_range)
{
    if(poses_database.size() < root_index + 1)
    {
        //printf("get_driftfree_poses: pose database do not have the root index! This should only happen in initializaion!\n");
        return;
    }
    vector<int> this_level;
    vector<int> next_level;
    this_level.push_back(root_index);
    driftfree_poses.push_back(root_index);
    // get the drift
    for(int i = 1; i < driftfree_range; i++)
    {
        for(auto this_it = this_level.begin(); this_it != this_level.end(); this_it++)
        {
            for(auto linked_it = poses_database[*this_it].linked_pose_index.begin(); 
                linked_it != poses_database[*this_it].linked_pose_index.end();
                linked_it++)
            {
                bool already_saved = (find(driftfree_poses.begin(), driftfree_poses.end(), *linked_it) != driftfree_poses.end());
                if(!already_saved)
                {
                    next_level.push_back(*linked_it);
                    driftfree_poses.push_back(*linked_it);
                }
            }
        }
        this_level.swap(next_level);
        next_level.clear();
    }
}
