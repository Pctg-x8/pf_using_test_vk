#version 450

layout(location = 0) in vec2 pos_in;
out gl_PerVertex { vec4 gl_Position; };

layout(push_constant) uniform ScreenInfo { layout(offset = 0) vec2 size; };

void main()
{
    gl_Position = vec4((pos_in.xy * 2.0 / size) * vec2(1.0, -1.0), 0.0, 1.0); /*divide by text-dpi(96)*/
}
