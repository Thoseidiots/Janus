// PBR shader source constants — Cook-Torrance BRDF with GGX NDF, Smith geometry, Fresnel-Schlick.
// All shader source is original code embedded as static strings.

pub const PBR_VERT_GLSL: &str = r#"
#version 450

layout(location = 0) in vec3 a_position;
layout(location = 1) in vec3 a_normal;
layout(location = 2) in vec2 a_uv;

layout(set = 0, binding = 0) uniform PerFrame {
    mat4 u_view;
    mat4 u_proj;
};

layout(set = 1, binding = 0) uniform PerObject {
    mat4 u_model;
};

layout(location = 0) out vec3 v_world_pos;
layout(location = 1) out vec3 v_normal;
layout(location = 2) out vec2 v_uv;

void main() {
    vec4 world_pos = u_model * vec4(a_position, 1.0);
    v_world_pos = world_pos.xyz;
    v_normal = normalize(mat3(transpose(inverse(u_model))) * a_normal);
    v_uv = a_uv;
    gl_Position = u_proj * u_view * world_pos;
}
"#;

pub const PBR_FRAG_GLSL: &str = r#"
#version 450

layout(location = 0) in vec3 v_world_pos;
layout(location = 1) in vec3 v_normal;
layout(location = 2) in vec2 v_uv;

layout(location = 0) out vec4 out_color;

layout(set = 0, binding = 1) uniform PerFrameLight {
    vec3  u_cam_pos;
    int   u_light_count;
    vec4  u_light_positions[8];   // xyz = position, w = unused
    vec4  u_light_colors[8];      // xyz = color * intensity, w = unused
};

layout(set = 2, binding = 0) uniform Material {
    vec4  u_base_color;   // xyz = albedo, w = alpha
    float u_metallic;
    float u_roughness;
    float u_ao;
    float _pad;
};

const float PI = 3.14159265358979323846;

// GGX / Trowbridge-Reitz Normal Distribution Function
float ndf_ggx(float NdotH, float roughness) {
    float a  = roughness * roughness;
    float a2 = a * a;
    float d  = NdotH * NdotH * (a2 - 1.0) + 1.0;
    return a2 / (PI * d * d);
}

// Schlick-GGX geometry term (single direction)
float geo_schlick_ggx(float NdotV, float roughness) {
    float r = roughness + 1.0;
    float k = (r * r) / 8.0;
    return NdotV / (NdotV * (1.0 - k) + k);
}

// Smith geometry function (both view and light directions)
float geo_smith(float NdotV, float NdotL, float roughness) {
    return geo_schlick_ggx(NdotV, roughness) * geo_schlick_ggx(NdotL, roughness);
}

// Fresnel-Schlick approximation
vec3 fresnel_schlick(float cos_theta, vec3 F0) {
    return F0 + (1.0 - F0) * pow(clamp(1.0 - cos_theta, 0.0, 1.0), 5.0);
}

void main() {
    vec3 albedo    = u_base_color.rgb;
    float metallic  = u_metallic;
    float roughness = max(u_roughness, 0.04); // avoid division by zero
    float ao        = u_ao;

    vec3 N = normalize(v_normal);
    vec3 V = normalize(u_cam_pos - v_world_pos);

    // Reflectance at normal incidence; interpolate between dielectric (0.04) and metal (albedo)
    vec3 F0 = mix(vec3(0.04), albedo, metallic);

    vec3 Lo = vec3(0.0);

    for (int i = 0; i < u_light_count; ++i) {
        vec3 L_dir    = u_light_positions[i].xyz - v_world_pos;
        float dist    = length(L_dir);
        vec3 L        = normalize(L_dir);
        vec3 H        = normalize(V + L);

        float attenuation = 1.0 / (dist * dist);
        vec3  radiance    = u_light_colors[i].xyz * attenuation;

        float NdotV = max(dot(N, V), 0.0001);
        float NdotL = max(dot(N, L), 0.0);
        float NdotH = max(dot(N, H), 0.0);
        float HdotV = max(dot(H, V), 0.0);

        // Cook-Torrance specular BRDF
        float D = ndf_ggx(NdotH, roughness);
        float G = geo_smith(NdotV, NdotL, roughness);
        vec3  F = fresnel_schlick(HdotV, F0);

        vec3 numerator   = D * G * F;
        float denominator = 4.0 * NdotV * NdotL + 0.0001;
        vec3 specular    = numerator / denominator;

        // Energy conservation: kS = F, kD = (1 - kS) * (1 - metallic)
        vec3 kD = (vec3(1.0) - F) * (1.0 - metallic);

        // Lambertian diffuse
        vec3 diffuse = kD * albedo / PI;

        Lo += (diffuse + specular) * radiance * NdotL;
    }

    // Ambient (simple IBL approximation)
    vec3 ambient = vec3(0.03) * albedo * ao;
    vec3 color   = ambient + Lo;

    // Reinhard tone mapping
    color = color / (color + vec3(1.0));
    // Gamma correction
    color = pow(color, vec3(1.0 / 2.2));

    out_color = vec4(color, u_base_color.a);
}
"#;

pub const PBR_WGSL: &str = r#"
// WGSL stub — PBR Cook-Torrance (to be expanded)
struct VertexOut {
    @builtin(position) position: vec4<f32>,
    @location(0) world_pos: vec3<f32>,
    @location(1) normal: vec3<f32>,
    @location(2) uv: vec2<f32>,
};

@vertex
fn vs_main(@location(0) pos: vec3<f32>,
           @location(1) normal: vec3<f32>,
           @location(2) uv: vec2<f32>) -> VertexOut {
    var out: VertexOut;
    out.position = vec4<f32>(pos, 1.0);
    out.world_pos = pos;
    out.normal = normal;
    out.uv = uv;
    return out;
}

@fragment
fn fs_main(in: VertexOut) -> @location(0) vec4<f32> {
    return vec4<f32>(1.0, 0.0, 1.0, 1.0); // stub: magenta
}
"#;

pub const PBR_HLSL: &str = r#"
// HLSL stub — PBR Cook-Torrance (to be expanded)
struct VSInput {
    float3 position : POSITION;
    float3 normal   : NORMAL;
    float2 uv       : TEXCOORD0;
};

struct PSInput {
    float4 position  : SV_POSITION;
    float3 world_pos : TEXCOORD0;
    float3 normal    : TEXCOORD1;
    float2 uv        : TEXCOORD2;
};

PSInput VSMain(VSInput input) {
    PSInput output;
    output.position  = float4(input.position, 1.0f);
    output.world_pos = input.position;
    output.normal    = input.normal;
    output.uv        = input.uv;
    return output;
}

float4 PSMain(PSInput input) : SV_TARGET {
    return float4(1.0f, 0.0f, 1.0f, 1.0f); // stub: magenta
}
"#;

pub const PBR_MSL: &str = r#"
// MSL stub — PBR Cook-Torrance (to be expanded)
#include <metal_stdlib>
using namespace metal;

struct VertexIn {
    float3 position [[attribute(0)]];
    float3 normal   [[attribute(1)]];
    float2 uv       [[attribute(2)]];
};

struct VertexOut {
    float4 position [[position]];
    float3 world_pos;
    float3 normal;
    float2 uv;
};

vertex VertexOut vertex_main(VertexIn in [[stage_in]]) {
    VertexOut out;
    out.position  = float4(in.position, 1.0);
    out.world_pos = in.position;
    out.normal    = in.normal;
    out.uv        = in.uv;
    return out;
}

fragment float4 fragment_main(VertexOut in [[stage_in]]) {
    return float4(1.0, 0.0, 1.0, 1.0); // stub: magenta
}
"#;

/// Solid magenta fallback shader — displayed when a shader fails to compile.
pub const FALLBACK_FRAG_GLSL: &str = r#"
#version 450

layout(location = 0) out vec4 out_color;

void main() {
    out_color = vec4(1.0, 0.0, 1.0, 1.0); // solid magenta error indicator
}
"#;
