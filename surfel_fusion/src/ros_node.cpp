#include <ros/ros.h>
#include <sensor_msgs/Image.h>
#include <sensor_msgs/PointCloud.h>
#include <geometry_msgs/PoseStamped.h>
#include <nav_msgs/Path.h>
#include <nav_msgs/Odometry.h>

#include <message_filters/subscriber.h>
#include <message_filters/synchronizer.h>
#include <message_filters/sync_policies/exact_time.h>
#include <message_filters/sync_policies/approximate_time.h>

typedef message_filters::sync_policies::ExactTime<sensor_msgs::PointCloud, nav_msgs::Path, nav_msgs::Odometry> exact_policy;

#include <surfel_map.h>

int main(int argc, char **argv)
{
  ros::init(argc, argv, "surfel_fusion");
  ros::NodeHandle nh("~");

  SurfelMap surfel_map(nh);

  ros::Subscriber sub_image = nh.subscribe("image", 5000, &SurfelMap::image_input, &surfel_map);
  ros::Subscriber sub_depth = nh.subscribe("depth", 5000, &SurfelMap::depth_input, &surfel_map);
  ros::Subscriber sub_save_map = nh.subscribe("save_map", 5000, &SurfelMap::save_map, &surfel_map);

  message_filters::Subscriber<sensor_msgs::PointCloud> sub_loop_stamps(nh, "loop_stamps", 1000);
  message_filters::Subscriber<nav_msgs::Path> sub_loop_path(nh, "loop_path", 1000);
  message_filters::Subscriber<nav_msgs::Odometry> sub_this_pose(nh, "this_pose", 1000);
  message_filters::Synchronizer<exact_policy> sync(exact_policy(1000), sub_loop_stamps, sub_loop_path, sub_this_pose);
  sync.registerCallback(boost::bind(&SurfelMap::orb_results_input, &surfel_map, _1, _2, _3));

  // ros::spin();

  // ros::Rate r(20);

  while(ros::ok())
  {
    ros::spinOnce();
  }

  string save_name;
  if(nh.getParam("save_name", save_name))
  {
    string pcd_name = save_name + ".PCD";
    string mesh_name = save_name + "_mesh.PLY";
    surfel_map.save_cloud(pcd_name);
    surfel_map.save_mesh(mesh_name);
  }

  return EXIT_SUCCESS;
}