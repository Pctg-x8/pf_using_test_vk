#version 450

layout(location = 0) in vec2 uv_in;
layout(location = 1) flat in int dir;
layout(location = 0) out vec4 target;

// bool curvein(vec2 uv) { return (pow(uv.x, 2.0) - uv.y) <= 0; }
void main()
{
    /*vec2 screen_scale = vec2(640, 360);
    if(!curvein(uv_in + 0.5 / screen_scale)) discard;
    target = vec4(1.0, 1.0, 1.0, 1.0);*/
    
    /*
    // gradients
    vec2 px = dFdx(uv_in), py = dFdy(uv_in);
    // chain rule
    float fx = (2 * uv_in.x) * px.x - px.y, fy = (2 * uv_in.x) * py.x - py.y;
    // signed distance
    float sd = (pow(uv_in.x, 2) - uv_in.y) / sqrt(pow(fx, 2) + pow(fy, 2));
    // linear alpha 1..inside, 0..outside
    float alpha = min(0.5 - sd, 1);
    if(alpha < 0) discard;
    target = vec4(1.0, 1.0, 1.0, 1.0) * alpha;
    */

    float sd = pow(uv_in.x, 2) - uv_in.y;
    if(sd * dir < 0) discard;
    target = vec4(1.0, 1.0, 1.0, 1.0);
}
