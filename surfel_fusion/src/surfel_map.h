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
#include <cerebro/LoopEdge.h>

#include <elements.h>
#include <fusion_functions.h>
// #include <parameters.h>
// #include <opengl_render/render_tool.h>

typedef pcl::PointXYZI PointType;
typedef pcl::PointCloud<PointType> PointCloud;

using namespace std;

struct PoseElement{
    //pose_index is the index in the vector in the database
    vector<SurfelElement> attached_surfels;
    geometry_msgs::Pose cam_pose;
    geometry_msgs::Pose loop_pose;
    vector<int> linked_pose_index;
    int points_begin_index;
    int points_pose_index;
    int world_id;
    int set_id;
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
    void path_input(const nav_msgs::PathConstPtr &loop_path_input);
    void loop_info_input(const cerebro::LoopEdgeConstPtr &loop_info_input);
    void extrinsic_input(const nav_msgs::OdometryConstPtr &ex_input);
    void orb_results_input(
        const sensor_msgs::PointCloudConstPtr &loop_stamp_input,
        const nav_msgs::PathConstPtr &loop_path_input,
        const nav_msgs::OdometryConstPtr &this_pose_input);
    void save_cloud(string save_path_name);
    void save_mesh(string save_path_name);
    void save_map(const std_msgs::StringConstPtr &save_map_input);
    void publish_all_pointcloud(ros::Time pub_stamp);

    // void loop_path_input(const nav_msgs::PathConstPtr &loop_path_input);
    // void loop_stamp_input(const sensor_msgs::PointcloudConstPtr &loop_stamp_input);
    // void this_pose_input(const nav_msgs::OdometryConstPtr &this_pose_input);

  private:
    void synchronize_msgs();
    // void initialize_map(cv::Mat image, cv::Mat depth, geometry_msgs::Pose pose, ros::Time stamp);
    // void fuse_map(cv::Mat image, cv::Mat depth, geometry_msgs::Pose pose, ros::Time stamp);
    void fuse_map(cv::Mat image, cv::Mat depth, Eigen::Matrix4f pose_input, int reference_index);
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
    void get_world_set_id(const string &str, int &world_id, int &set_id)
    {
        char separator = ':';
        string world_id_str;
        string set_id_str;
        std::string::const_iterator start = str.begin();
        std::string::const_iterator end = str.end();
        std::string::const_iterator sep0 = std::find(start, end, separator);
        std::string::const_iterator sep1 = std::find(sep0+1, end, separator);
        std::string::const_iterator sep2 = std::find(sep1+1, end, separator);
        world_id_str = std::string(sep0+1, sep1);
        set_id_str = std::string(sep2+1, end);
        world_id = std::stoi(world_id_str);
        set_id = std::stoi(set_id_str);
    }

    void render_depth(geometry_msgs::Pose &pose);
    void publish_neighbor_pointcloud(ros::Time pub_stamp, int reference_index);
    void publish_this_set_pointcloud(ros::Time pub_stamp, int reference_index);
    void publish_raw_pointcloud(cv::Mat &depth, cv::Mat &reference, geometry_msgs::Pose &pose);
    void publish_pose_graph(ros::Time pub_stamp, int reference_index);
    void calculate_memory_usage();

    // for surfel save into mesh
    void push_a_surfel(vector<float> &vertexs, SurfelElement &this_surfel);

    // receive buffer
    std::deque<std::pair<ros::Time, cv::Mat>> image_buffer;
    std::deque<std::pair<ros::Time, cv::Mat>> depth_buffer;
    std::deque<std::pair<ros::Time, int>> pose_reference_buffer;
    std::vector<std::pair<double, double>> linked_pose_stamps;

    // geometry_msgs::PoseStamped await_pose;
    // std::list<std::pair<ros::Time, geometry_msgs::Pose> > pose_buffer;

    // render tool
    // RenderTool render_tool;

    // camera param
    int cam_width;
    int cam_height;
    float cam_fx, cam_fy, cam_cx, cam_cy;
    Eigen::Matrix4d extrinsic_matrix;
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
    std::vector<int> pointcloud_pose_index;

    // ros related
    ros::NodeHandle &nh;
    ros::Publisher pointcloud_publish;
    ros::Publisher set_pointcloud_publish;
    ros::Publisher raw_pointcloud_publish;
    ros::Publisher loop_path_publish;
    ros::Publisher driftfree_path_publish;
    ros::Publisher loop_marker_publish;
    ros::Publisher cam_pose_publish;

    // for gaofei experiment
    bool is_first_path;
    double pre_path_delete_time;
};