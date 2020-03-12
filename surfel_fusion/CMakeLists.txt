cmake_minimum_required(VERSION 2.8.3)
project(surfel_fusion)

#set(CUDA_USE_STATIC_CUDA_RUNTIME OFF)
# set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "/home/uav/catkin_ws/src/surfel_fusion/cmake")
set(CMAKE_BUILD_TYPE "Release")
set(CMAKE_CXX_FLAGS "-std=c++11 -DEIGEN_DONT_PARALLELIZE -pthread ${OpenMP_CXX_FLAGS}")
set(CMAKE_CXX_FLAGS_RELEASE "-O3 -Wall -g")
set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} ${OpenMP_EXE_LINKER_FLAGS}")

find_package(catkin REQUIRED COMPONENTS
    roscpp
    cv_bridge
    sensor_msgs
    nav_msgs
    message_filters
    pcl_ros
    )
catkin_package(
)

find_package(Boost REQUIRED COMPONENTS system filesystem)
find_package(Eigen3 REQUIRED)
include_directories(${EIGEN3_INCLUDE_DIR})
find_package(OpenCV REQUIRED)
#find_package(librealsense2 EXACT REQUIRED PATHS /usr/local/include/librealsense2/)
# find_package(OpenGL REQUIRED)
# find_package(glfw3 REQUIRED)
# find_package(GLEW REQUIRED)
# find_package(glm REQUIRED)
#find_package(CUDA REQUIRED)
# find_package(OpenMP REQUIRED)
#find_package(PCL REQUIRED)

# set(CUDA_NVCC_FLAGS
#     ${CUDA_NVCC_FLAGS};
#    -O3 -gencode arch=compute_61,code=sm_61  # gtx 980 Ti / 1080
#     )


include_directories(
  src
  ${catkin_INCLUDE_DIRS}
  ${EIGEN3_INCLUDE_DIR}
  ${Boost_INCLUDE_DIRS}
  # ${GLEW_INCLUDE_DIR}
  # ${GLM_INCLUDE_DIR}
  # ${OPENGL_INCLUDE_DIR}
  # ${glfw3_DIR}
#  ${PCL_INCLUDE_DIRS}
)

add_executable(surfel_fusion
    src/ros_node.cpp
    src/surfel_map.cpp
    # src/cuda_functions.cu
    src/fusion_functions.cpp 
    # src/opengl_render/render_tool.cpp
    # src/opengl_render/shader.cpp
    )
target_link_libraries(surfel_fusion 
  ${catkin_LIBRARIES} 
  ${OpenCV_LIBS} 
  ${Boost_LIBRARIES} 
  # ${OPENGL_LIBRARY}
	# ${GLEW_LIBRARY}
  # ${GLFW3_LIBRARY}
  # -fopenmp
  )