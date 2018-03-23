#version 450

layout(location = 0) in vec2 pos_in;
layout(location = 1) in ivec4 loop_blinn_data;
out gl_PerVertex { vec4 gl_Position; };
layout(location = 0) out vec2 uv_out;
layout(location = 1) flat out int lb_dir;

layout(push_constant) uniform ScreenInfo { layout(offset = 0) vec2 size; };

void main()
{
    gl_Position = vec4((2.0 * pos_in / size) * vec2(1.0, -1.0), 0.0, 1.0);
    uv_out = vec2(loop_blinn_data.xy / 2.0);
    lb_dir = loop_blinn_data.z;
}
