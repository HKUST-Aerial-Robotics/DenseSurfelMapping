#include "opengl_render/render_tool.h"
#include <string>
// opencv
#include <opencv2/opencv.hpp>

void RenderTool::initialize_rendertool(int width_, int height_, float fx_, float fy_, float cx_, float cy_)
{
    width = width_;
    height = height_;
    fx = fx_;
    fy = fy_;
    cx = cx_;
    cy = cy_;

    printf("initializing the RenderTool ... \n");
    printf("windex size (%dx%d).\n", width, height);
    printf("camera parameter fx: %f, fy: %f, cx: %f, cy: %f.\n", fx, fy, cx, cy);

    if (!glfwInit())
    {
        fprintf(stderr, "Failed to initialize GLFW\n");
    }

    printf("initializing the RenderTool: window \n");
    glfwWindowHint(GLFW_SAMPLES, 4);
    glfwWindowHint(GLFW_VISIBLE, GL_FALSE);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3);
    glfwWindowHint(GLFW_OPENGL_FORWARD_COMPAT, GL_TRUE);// To make MacOS happy; should not be needed
    glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);

    // Open a window and create its OpenGL context
    window = glfwCreateWindow(width, height, "RenderTool", NULL, NULL);
    if (window == NULL)
    {
        fprintf(stderr, "Failed to open GLFW window. If you have an Intel GPU, they are not 3.3 compatible. Try the 2.1 version of the tutorials.\n");
        glfwTerminate();
    }
    glfwMakeContextCurrent(window);

    // Initialize GLEW
    glewExperimental = true; // Needed for core profile
    if (glewInit() != GLEW_OK)
    {
        fprintf(stderr, "Failed to initialize GLEW\n");
        glfwTerminate();
    }

    // Dark blue background
    glClearColor(0.0f, 0.0f, 0.0f, 0.0f);
    // Enable depth test
    glEnable(GL_DEPTH_TEST);
    // Accept fragment if it closer to the camera than the former one
    glDepthFunc(GL_LESS);

    glGenVertexArrays(1, &VertexArrayID);
    glBindVertexArray(VertexArrayID);

    printf("initializing the RenderTool: frame buffer \n");
    FramebufferName = 0;
    glGenFramebuffers(1, &FramebufferName);
    glBindFramebuffer(GL_FRAMEBUFFER, FramebufferName);
    glGenTextures(1, &renderedTexture);
    glBindTexture(GL_TEXTURE_2D, renderedTexture);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB32F, width, height, 0, GL_RGB, GL_FLOAT, 0);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);

    // The depth buffer
    glGenRenderbuffers(1, &depthrenderbuffer);
    glBindRenderbuffer(GL_RENDERBUFFER, depthrenderbuffer);
    glRenderbufferStorage(GL_RENDERBUFFER, GL_DEPTH_COMPONENT, width, height);
    glFramebufferRenderbuffer(GL_FRAMEBUFFER, GL_DEPTH_ATTACHMENT, GL_RENDERBUFFER, depthrenderbuffer);

    // Set "renderedTexture" as our colour attachement #0
    glFramebufferTexture(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, renderedTexture, 0);

    // Set the list of draw buffers.
    DrawBuffers[0] = {GL_COLOR_ATTACHMENT0};
    glDrawBuffers(1, DrawBuffers); // "1" is the size of DrawBuffers

    printf("initializing the RenderTool: shader \n");
    std::string fold_path = "/home/uav/catkin_ws/src/surfel_fusion/src/opengl_render/";
    std::string vert_name = "index_render.vert";
    std::string geom_name = "index_render.geom";
    std::string frag_name = "index_render.frag";
    RenderIndexProgramID = LoadShaders(
        (fold_path + vert_name).c_str(),
        (fold_path + geom_name).c_str(),
        (fold_path + frag_name).c_str());
    // glUseProgram(RenderIndexProgramID);
    printf("initializing the RenderTool: GLuint \n");
    CameraProjectID = glGetUniformLocation(RenderIndexProgramID, "Project");
    CameraPoseID = glGetUniformLocation(RenderIndexProgramID, "Pose");
    glGenBuffers(1, &PositionBuffer);
    glGenBuffers(1, &NormRBuffer);

    printf("initializing the RenderTool: camera matrix \n");
    Perspective = glm::perspective(glm::radians(90.0f), (float)width / (float)height, 0.01f, 100.0f);
    Perspective[0][0] = fx / width * 2.0;
    Perspective[2][0] = cx / width * 2.0 - 1.0;
    Perspective[1][1] = fy / height * -2.0;
    Perspective[2][1] = (cy / height * 2.0 - 1.0) * -1.0;
}

RenderTool::~RenderTool()
{
    glDeleteProgram(RenderIndexProgramID);
    glDeleteBuffers(1, &PositionBuffer);
    glDeleteBuffers(1, &NormRBuffer);
    glDeleteVertexArrays(1, &VertexArrayID);
    glfwTerminate();
}

void RenderTool::render_surfels(
    std::vector<float> &position,
    std::vector<float> &normr,
    std::vector<float> &depth_map,
    Eigen::Matrix4f &camera_in_world)
{
    glUseProgram(RenderIndexProgramID);
    int surfel_number = position.size() / 3;
    std::cout << "render tool: need to render: " << surfel_number << std::endl;
    glBindBuffer(GL_ARRAY_BUFFER, PositionBuffer);
    glBufferData(GL_ARRAY_BUFFER, sizeof(float) * position.size(), position.data(), GL_STREAM_DRAW);
    glBindBuffer(GL_ARRAY_BUFFER, NormRBuffer);
    glBufferData(GL_ARRAY_BUFFER, sizeof(float) * normr.size(), normr.data(), GL_STREAM_DRAW);

    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

    glUniformMatrix4fv(CameraProjectID, 1, GL_FALSE, &Perspective[0][0]);

    // set camera position
    glm::vec3 c_position = glm::vec3(
        camera_in_world(0, 3),
        camera_in_world(1, 3),
        camera_in_world(2, 3));
    glm::vec3 c_lookat = glm::vec3(
        camera_in_world(0, 2),
        camera_in_world(1, 2),
        camera_in_world(2, 2));
    glm::vec3 c_headat = glm::vec3(
        camera_in_world(0, 1) * -1.0f,
        camera_in_world(1, 1) * -1.0f,
        camera_in_world(2, 1) * -1.0f);
    glm::mat4 Pose = glm::lookAt(
        c_position,
        c_position + c_lookat,
        c_headat);
    glUniformMatrix4fv(CameraPoseID, 1, GL_FALSE, &Pose[0][0]);

    //attribute buffer
    glEnableVertexAttribArray(0);
    glBindBuffer(GL_ARRAY_BUFFER, PositionBuffer);
    glVertexAttribPointer(
        0,        // attribute 0.
        3,        // size
        GL_FLOAT, // type
        GL_FALSE, // normalized?
        0,        // stride
        (void *)0 // array buffer offset
    );
    glEnableVertexAttribArray(1);
    glBindBuffer(GL_ARRAY_BUFFER, NormRBuffer);
    glVertexAttribPointer(
        1,        // attribute 1.
        4,        // size
        GL_FLOAT, // type
        GL_FALSE, // normalized?
        0,        // stride
        (void *)0 // array buffer offset
    );

    glDrawArrays(GL_POINTS, 0, surfel_number);
    if(depth_map.size() != width*height)
        depth_map.resize(width*height);
    // glReadPixels(0, 0, width, height, GL_RED, GL_FLOAT, depth_map.data());
    glReadPixels(0, 0, width, height, GL_GREEN, GL_FLOAT, depth_map.data());
    glDisableVertexAttribArray(0);
    glDisableVertexAttribArray(1);
}