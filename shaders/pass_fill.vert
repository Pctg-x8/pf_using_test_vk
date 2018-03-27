#version 450

layout(location = 0) in vec2 pos_in;
out gl_PerVertex { vec4 gl_Position; };

layout(set = 0, binding = 0) uniform samplerBuffer glyphTransformSTExt;
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
void main()
{
    vec4 gtst; vec2 gtext; fetchGlyphTransformFor(glyph_index, gtst, gtext);
    gl_Position = vec4((2.0 * transformAffine(pos_in, gtst, gtext) / size) * vec2(1.0, -1.0), 0.0, 1.0);
}
