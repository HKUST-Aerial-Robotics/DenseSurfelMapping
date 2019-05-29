#version 330 core

// Ouput data
// in vec3 position;
in float true_depth;
flat in int index;
// out vec3 color;

out vec3 color2;

void main()
{
    color2 = vec3(index, -1.0*true_depth, 0.0);
    // color2 = vec3(true_depth*-1.0, index, 0.0);
}