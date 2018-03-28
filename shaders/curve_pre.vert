#version 450

layout(location = 0) in vec2 pos_in;
layout(location = 1) in ivec4 loop_blinn_data;
out gl_PerVertex { vec4 gl_Position; };
layout(location = 0) out vec2 uv_out;
layout(location = 1) flat out int lb_dir;

layout(set = 0, binding = 0) uniform samplerBuffer glyphTransformSTExt;
layout(set = 0, binding = 1) uniform HintConst { vec4 hint; };
layout(push_constant) uniform ScreenInfo
{
    layout(offset = 0) vec2 size; layout(offset = 4 * 2) int glyph_index;
};

void fetchGlyphTransformFor(int id, out vec4 st, out vec2 ext)
{
    st  = texelFetch(glyphTransformSTExt, id * 2 + 0);
    ext = texelFetch(glyphTransformSTExt, id * 2 + 1).xy;
}
vec2 transformAffine(vec2 vin, vec4 st, vec2 ext) { return vin * st.xy + vin.yx * ext + st.zw; }
vec2 applyHints(vec2 pos)
{
    if(pos.y >= hint.z) { pos.y -= hint.z - hint.w; }
    else if(pos.y >= hint.x) { pos.y = mix(hint.y, hint.w, (pos.y - hint.x) / (hint.z - hint.x)); }
    else if(pos.y >= 0.0) { pos.y = mix(0.0, hint.y, pos.y / hint.x); }

    return pos;
}
void main()
{
    vec4 gtst; vec2 gtext; fetchGlyphTransformFor(glyph_index, gtst, gtext);
    gl_Position = vec4((2.0 * transformAffine(applyHints(pos_in), gtst, gtext) / size) * vec2(1.0, -1.0), 0.0, 1.0);
    uv_out = vec2(loop_blinn_data.xy / 2.0);
    lb_dir = loop_blinn_data.z;
}
