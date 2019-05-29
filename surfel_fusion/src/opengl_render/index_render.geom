#version 330 core

layout(points) in;
layout(triangle_strip, max_vertices = 6) out;

// in vec3 vPosition[];
// in mat4 vP[];
// in mat4 vPose[];
// flat in float vRadius[];
in int gl_PrimitiveID;
in mat4 vProject[];
in mat4 vPose[];
in vec3 vPosition[];
in vec4 vNormr[];

// out vec3 position;
out float true_depth;
flat out int index;
void main()
{
    // calculate the vertexs
    vec3 surfel_norm = vNormr[0].xyz;
    vec3 x_dir = normalize(vec3(-surfel_norm.y, surfel_norm.x, 0.0));
    vec3 y_dir = cross(surfel_norm, x_dir);
    float radius = vNormr[0].w;
    float h_r = radius * 0.5;
    float t_r = radius * 0.86603;

    // siz points
    index = gl_PrimitiveID;
    // position = vPosition[0];
    vec4 space_position = vPose[0] * vec4(vPosition[0] - x_dir * h_r - y_dir * t_r, 1.0);
    gl_Position = vProject[0] * space_position;
    true_depth = space_position.z;
    EmitVertex();
    space_position = vPose[0] * vec4(vPosition[0] + x_dir * h_r - y_dir * t_r, 1.0);
    gl_Position = vProject[0] * space_position;
    true_depth = space_position.z;
    EmitVertex();
    space_position = vPose[0] * vec4(vPosition[0] - x_dir * radius, 1.0);
    gl_Position = vProject[0] * space_position;
    true_depth = space_position.z;
    EmitVertex();
    space_position = vPose[0] * vec4(vPosition[0] + x_dir * radius, 1.0);
    gl_Position = vProject[0] * space_position;
    true_depth = space_position.z;
    EmitVertex();
    space_position = vPose[0] * vec4(vPosition[0] - x_dir * h_r + y_dir * t_r, 1.0);
    gl_Position = vProject[0] * space_position;
    true_depth = space_position.z;
    EmitVertex();
    space_position = vPose[0] * vec4(vPosition[0] + x_dir * h_r + y_dir * t_r, 1.0);
    gl_Position = vProject[0] * space_position;
    true_depth = space_position.z;
    EmitVertex();
}
