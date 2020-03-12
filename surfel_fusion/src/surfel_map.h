#include <list>
#include <vector>
#include <set>
#include <iostream>
#include <chrono>
#include <thread>
#include <Eigen/Eigen>
#include <opencv2/opencv.hpp>

#include <boost/algorithm/string.hpp>
#include <boost/interprocess/sync/file_lock.hpp>
#include <pcl_ros/point_cloud.h>
#include <pcl/filters/passthrough.h>
#include <pcl/point_types.h>

#include <boost/shared_ptr.hpp>

#include <ros/ros.h>
#include <cv_bridge/cv_bridge.h>
#include <sensor_msgs/image_encodings.h>
#include <sensor_msgs/Image.h>
#include <sensor_msgs/PointCloud.h>
#include <geometry_msgs/Pose.h>
#include <std_msgs/String.h>
#include <geometry_msgs/PoseStamped.h>
#include <geometry_msgs/PointStamped.h>
#include <nav_msgs/Path.h>
#include <nav_msgs/Odometry.h>
#include <visualization_msgs/Marker.h>
#include <std_srvs/SetBool.h>
#include <std_msgs/Int16.h>

#include <elements.h>
#include <fusion_functions.h>
// #include <parameters.h>
// #include <opengl_render/render_tool.h>

typedef pcl::PointXYZI PointType;
typedef pcl::PointXYZRGBNormal RgbPointType;
typedef pcl::PointCloud<PointType> PointCloud;
typedef pcl::PointCloud<RgbPointType> RgbPointCloud;

using namespace std;

struct PoseElement{
    //pose_index is the index in the vector in the database
    vector<SurfelElement> attached_surfels;
    geometry_msgs::Pose cam_pose;
    geometry_msgs::Pose loop_pose;
    vector<int> linked_pose_index;
    int points_begin_index;
    int points_pose_index;
    ros::Time cam_stamp;
    PoseElement() : points_begin_index(-1), points_pose_index(-1) {}
};

class SurfelMap
{
public:
    SurfelMap(ros::NodeHandle &_nh);
    ~SurfelMap();

    void image_input(const sensor_msgs::ImageConstPtr &image_input);
    void depth_input(const sensor_msgs::ImageConstPtr &image_input);
    void color_input(const sensor_msgs::ImageConstPtr &image_input);
    void path_input(const nav_msgs::PathConstPtr &loop_path_input);
    void extrinsic_input(const nav_msgs::OdometryConstPtr &ex_input);
    void surfel_cmd_callback(const std_msgs::Int16ConstPtr &cmd);
    void orb_results_input(
        const sensor_msgs::PointCloudConstPtr &loop_stamp_input,
        const nav_msgs::PathConstPtr &loop_path_input,
        const nav_msgs::OdometryConstPtr &this_pose_input);
    void save_cloud(string save_path_name);
    void save_mesh(string save_path_name);
    void save_map(const std_msgs::StringConstPtr &save_map_input);
    void publish_all_pointcloud(ros::Time pub_stamp);

    bool surfel_state;
    void set_map_dir(string str);

    // void loop_path_input(const nav_msgs::PathConstPtr &loop_path_input);
    // void loop_stamp_input(const sensor_msgs::PointcloudConstPtr &loop_stamp_input);
    // void this_pose_input(const nav_msgs::OdometryConstPtr &this_pose_input);

  private:
    void synchronize_msgs();
    // void initialize_map(cv::Mat grey_image, cv::Mat depth, geometry_msgs::Pose pose, ros::Time stamp);
    // void fuse_map(cv::Mat grey_image, cv::Mat depth, geometry_msgs::Pose pose, ros::Time stamp);
    void fuse_map(cv::Mat image, cv::Mat depth, Eigen::Matrix4f pose_input, int reference_index);
    // void initialize_map(cv::Mat image, cv::Mat depth, geometry_msgs::Pose pose, ros::Time stamp);
    // void fuse_map(cv::Mat image, cv::Mat depth, geometry_msgs::Pose pose, ros::Time stamp);

    void move_add_surfels(int reference_index);
    void move_all_surfels();
    bool synchronize_buffer();
    void get_add_remove_poses(int root_index, vector<int> &pose_to_add, vector<int> &pose_to_remove);
    void get_driftfree_poses(int root_index, vector<int> &driftfree_poses, int driftfree_range);
    // void fuse_inputs();
    // void get_inactive_surfels();
    // void get_drift_poses(int root_index, vector<int> &drift_poses);

    void pose_ros2eigen(geometry_msgs::Pose &pose, Eigen::Matrix4d &T);
    void pose_eigen2ros(Eigen::Matrix4d &T, geometry_msgs::Pose &pose);

    void render_depth(geometry_msgs::Pose &pose);
    void publish_neighbor_pointcloud(ros::Time pub_stamp, int reference_index);
    void publish_raw_pointcloud(cv::Mat &depth, cv::Mat &reference, geometry_msgs::Pose &pose);
    void publish_pose_graph(ros::Time pub_stamp, int reference_index);
    void calculate_memory_usage();

    // for surfel save into mesh
    void push_a_surfel(vector<float> &vertexs, SurfelElement &this_surfel);

    // image
    cv::Mat debug_image;

    // receive buffer
    std::deque<std::pair<ros::Time, cv::Mat>> image_buffer;
    std::deque<std::pair<ros::Time, cv::Mat>> depth_buffer;
    std::deque<std::pair<ros::Time, cv::Mat>> color_buffer;
    std::deque<std::pair<ros::Time, int>> pose_reference_buffer;

    // geometry_msgs::PoseStamped await_pose;
    // std::list<std::pair<ros::Time, geometry_msgs::Pose> > pose_buffer;

    // render tool
    // RenderTool render_tool;

    // camera param
    int cam_width;
    int cam_height;
    float cam_fx, cam_fy, cam_cx, cam_cy;
    Eigen::Matrix4d extrinsic_matrix, T_d2c;
    bool extrinsic_matrix_initialized;

    Eigen::Matrix3d camera_matrix;
    Eigen::Matrix3d imu_cam_rot;
    Eigen::Vector3d imu_cam_tra;

    // fuse param
    float far_dist, near_dist;

    // fusion tools
    FusionFunctions fusion_functions;

    // gpu param
    // FuseParameters fuse_param;
    // FuseParameters *fuse_param_gpuptr;
    
    // database
    vector<SurfelElement> local_surfels;
    vector<PoseElement> poses_database;
    // Eigen::Matrix4d local_loop_warp;
    std::set<int> local_surfels_indexs;
    int drift_free_poses;

    // for inactive warp
    std::vector<std::thread> warp_thread_pool;
    int warp_thread_num;
    void warp_surfels();
    void warp_inactive_surfels_cpu_kernel(int thread_i, int step);
    void warp_active_surfels_cpu_kernel(int thread_i, int thread_num, Eigen::Matrix4f transform_m);

    // for fast publish
    PointCloud::Ptr inactive_pointcloud;
    RgbPointCloud::Ptr rgb_inactive_pcd;
    std::vector<int> pointcloud_pose_index;

    // ros related
    ros::NodeHandle &nh;
    ros::Publisher pointcloud_publish;
    ros::Publisher raw_pointcloud_publish;
    ros::Publisher rgb_pointcloud_publish;
    ros::Publisher loop_path_publish;
    ros::Publisher driftfree_path_publish;
    ros::Publisher loop_marker_publish;
    ros::Publisher cam_pose_publish;
    ros::Publisher sp_img_publish;

    // save map
    string map_dir;

    // for gaofei experiment
    bool is_first_path;
    double pre_path_delete_time;
};