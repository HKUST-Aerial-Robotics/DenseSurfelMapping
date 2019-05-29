#version 330 core

// Input vertex data, different for all executions of this shader.
layout(location = 0) in vec3 position;
layout(location = 1) in vec4 norm_r;

uniform mat4 Project;
uniform mat4 Pose;

out mat4 vProject;
out mat4 vPose;
out vec3 vPosition;
out vec4 vNormr;

void main()
{
    vProject = Project;
    vPose = Pose;
    vPosition = position;
    vNormr = norm_r;
}