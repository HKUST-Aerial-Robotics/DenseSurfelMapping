// Include GLEW
#include <GL/glew.h>
// Include GLFW
#include <GLFW/glfw3.h>
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>

#include <iostream>
#include <vector>
#include <cstdlib>
#include "opengl_render/shader.h"

#include <Eigen/Eigen>

class RenderTool
{
public:
    ~RenderTool();
    void initialize_rendertool(int width_, int height_, float fx_, float fy_, float cx_, float cy_);
    void render_surfels(
        std::vector<float> &position,
        std::vector<float> &normr,
        std::vector<float> &depth_map,
        Eigen::Matrix4f &camera_in_world);

  private:
    GLFWwindow *window;
    int width, height;
    float fx, fy, cx, cy;
    glm::mat4 Perspective;

    // to render the index map
    GLuint FramebufferName;
    GLuint renderedTexture;
    GLuint depthrenderbuffer;
    GLenum DrawBuffers[1];

    GLuint RenderIndexProgramID;
    GLuint VertexArrayID;
    GLuint CameraProjectID;
    GLuint CameraPoseID;
    GLuint PositionBuffer;
    GLuint NormRBuffer;
};