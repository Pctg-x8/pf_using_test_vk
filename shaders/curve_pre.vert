#version 450

layout(location = 0) in vec2 uv_in;
// 1, 2, 3で順番に消費するらしい
layout(location = 1) in vec4 pos_in[3];
out gl_PerVertex { vec4 gl_Position; };
layout(location = 0) out vec2 uv_out;

void main()
{
    gl_Position = pos_in[gl_VertexIndex] * 0.5f; gl_Position.w = 1.0;
    uv_out = uv_in;
}
