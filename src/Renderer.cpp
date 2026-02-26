// STRICT REQUIREMENT: Jolt.h must be included first
#include <Jolt/Jolt.h>
#include "Renderer.h"
#include "PhysicsCore.h" // We need this for the Dimensional Layers

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <iostream>
#include <vector>

#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtc/type_ptr.hpp>

#include <Jolt/Physics/Body/BodyInterface.h>
#include <Jolt/Physics/Collision/Shape/BoxShape.h>
#include <Jolt/Physics/Collision/Shape/Shape.h>
#include <Jolt/Physics/Collision/Shape/SphereShape.h>
#include <Jolt/Physics/Collision/Shape/CylinderShape.h>

extern Camera gCamera;

namespace
{
constexpr const char* kVertexShader = R"(
#version 330 core
layout (location = 0) in vec3 aPos;
layout (location = 1) in vec3 aNormal;

uniform mat4 uModel;
uniform mat4 uView;
uniform mat4 uProj;

out vec3 FragPos;
out vec3 Normal;
out vec3 ViewPos;

void main()
{
    vec4 world_pos = uModel * vec4(aPos, 1.0);
    FragPos = world_pos.xyz;
    Normal = mat3(transpose(inverse(uModel))) * aNormal;
    ViewPos = vec3(inverse(uView) * vec4(0.0, 0.0, 0.0, 1.0));
    gl_Position = uProj * uView * world_pos;
}
)";

constexpr const char* kFragmentShader = R"(
#version 330 core
out vec4 FragColor;

in vec3 FragPos;
in vec3 Normal;
in vec3 ViewPos;

struct Light {
    vec3 position;
    vec3 color;
    float intensity;
};

uniform vec3 uViewPos;
uniform vec3 uObjectColor;
uniform float uMetallic;
uniform float uRoughness;
uniform Light uLights[4];
uniform int uNumLights;

const float PI = 3.14159265359;

float DistributionGGX(vec3 N, vec3 H, float roughness) {
    float a = roughness * roughness;
    float a2 = a * a;
    float NdotH = max(dot(N, H), 0.0);
    float NdotH2 = NdotH * NdotH;
    float nom = a2;
    float denom = (NdotH2 * (a2 - 1.0) + 1.0);
    denom = PI * denom * denom;
    return nom / denom;
}

float GeometrySchlickGGX(float NdotV, float roughness) {
    float r = (roughness + 1.0);
    float k = (r * r) / 8.0;
    float nom = NdotV;
    float denom = NdotV * (1.0 - k) + k;
    return nom / denom;
}

float GeometrySmith(vec3 N, vec3 V, vec3 L, float roughness) {
    float NdotV = max(dot(N, V), 0.0);
    float NdotL = max(dot(N, L), 0.0);
    float ggx2 = GeometrySchlickGGX(NdotV, roughness);
    float ggx1 = GeometrySchlickGGX(NdotL, roughness);
    return ggx1 * ggx2;
}

vec3 fresnelSchlick(float cosTheta, vec3 F0) {
    return F0 + (1.0 - F0) * pow(clamp(1.0 - cosTheta, 0.0, 1.0), 5.0);
}

void main()
{
    vec3 N = normalize(Normal);
    vec3 V = normalize(uViewPos - FragPos);
    
    vec3 F0 = vec3(0.04);
    F0 = mix(F0, uObjectColor, uMetallic);
    
    vec3 Lo = vec3(0.0);
    
    for(int i = 0; i < uNumLights; ++i) {
        vec3 L = normalize(uLights[i].position - FragPos);
        vec3 H = normalize(V + L);
        float distance = length(uLights[i].position - FragPos);
        float attenuation = 1.0 / (distance * distance);
        vec3 radiance = uLights[i].color * uLights[i].intensity * attenuation;
        
        float NDF = DistributionGGX(N, H, uRoughness);
        float G = GeometrySmith(N, V, L, uRoughness);
        vec3 F = fresnelSchlick(max(dot(H, V), 0.0), F0);
        
        vec3 numerator = NDF * G * F;
        float denominator = 4.0 * max(dot(N, V), 0.0) * max(dot(N, L), 0.0) + 0.0001;
        vec3 specular = numerator / denominator;
        
        vec3 kS = F;
        vec3 kD = vec3(1.0) - kS;
        kD *= 1.0 - uMetallic;
        
        float NdotL = max(dot(N, L), 0.0);
        Lo += (kD * uObjectColor / PI + specular) * radiance * NdotL;
    }
    
    vec3 ambient = vec3(0.15) * uObjectColor;
    
    float fresnel = pow(1.0 - max(dot(N, V), 0.0), 3.0);
    vec3 reflection = mix(vec3(0.1), vec3(0.8), fresnel) * uMetallic;
    
    vec3 color = ambient + Lo + reflection;
    
    color = color / (color + vec3(1.0));
    color = pow(color, vec3(1.0/2.2));
    
    FragColor = vec4(color, 1.0);
}
)";

constexpr float kCubeVertices[] = {
    -0.5f, -0.5f, -0.5f,   0.0f,  0.0f, -1.0f,
     0.5f, -0.5f, -0.5f,   0.0f,  0.0f, -1.0f,
     0.5f,  0.5f, -0.5f,   0.0f,  0.0f, -1.0f,
     0.5f,  0.5f, -0.5f,   0.0f,  0.0f, -1.0f,
    -0.5f,  0.5f, -0.5f,   0.0f,  0.0f, -1.0f,
    -0.5f, -0.5f, -0.5f,   0.0f,  0.0f, -1.0f,

    -0.5f, -0.5f,  0.5f,   0.0f,  0.0f,  1.0f,
     0.5f, -0.5f,  0.5f,   0.0f,  0.0f,  1.0f,
     0.5f,  0.5f,  0.5f,   0.0f,  0.0f,  1.0f,
     0.5f,  0.5f,  0.5f,   0.0f,  0.0f,  1.0f,
    -0.5f,  0.5f,  0.5f,   0.0f,  0.0f,  1.0f,
    -0.5f, -0.5f,  0.5f,   0.0f,  0.0f,  1.0f,

    -0.5f,  0.5f,  0.5f,  -1.0f,  0.0f,  0.0f,
    -0.5f,  0.5f, -0.5f,  -1.0f,  0.0f,  0.0f,
    -0.5f, -0.5f, -0.5f,  -1.0f,  0.0f,  0.0f,
    -0.5f, -0.5f, -0.5f,  -1.0f,  0.0f,  0.0f,
    -0.5f, -0.5f,  0.5f,  -1.0f,  0.0f,  0.0f,
    -0.5f,  0.5f,  0.5f,  -1.0f,  0.0f,  0.0f,

     0.5f,  0.5f,  0.5f,   1.0f,  0.0f,  0.0f,
     0.5f,  0.5f, -0.5f,   1.0f,  0.0f,  0.0f,
     0.5f, -0.5f, -0.5f,   1.0f,  0.0f,  0.0f,
     0.5f, -0.5f, -0.5f,   1.0f,  0.0f,  0.0f,
     0.5f, -0.5f,  0.5f,   1.0f,  0.0f,  0.0f,
     0.5f,  0.5f,  0.5f,   1.0f,  0.0f,  0.0f,

    -0.5f, -0.5f, -0.5f,   0.0f, -1.0f,  0.0f,
     0.5f, -0.5f, -0.5f,   0.0f, -1.0f,  0.0f,
     0.5f, -0.5f,  0.5f,   0.0f, -1.0f,  0.0f,
     0.5f, -0.5f,  0.5f,   0.0f, -1.0f,  0.0f,
    -0.5f, -0.5f,  0.5f,   0.0f, -1.0f,  0.0f,
    -0.5f, -0.5f, -0.5f,   0.0f, -1.0f,  0.0f,

    -0.5f,  0.5f, -0.5f,   0.0f,  1.0f,  0.0f,
     0.5f,  0.5f, -0.5f,   0.0f,  1.0f,  0.0f,
     0.5f,  0.5f,  0.5f,   0.0f,  1.0f,  0.0f,
     0.5f,  0.5f,  0.5f,   0.0f,  1.0f,  0.0f,
    -0.5f,  0.5f,  0.5f,   0.0f,  1.0f,  0.0f,
    -0.5f,  0.5f, -0.5f,   0.0f,  1.0f,  0.0f,
};

void BuildSphereMesh(int stacks, int slices, std::vector<float>& vertices, std::vector<uint32_t>& indices)
{
    vertices.clear();
    indices.clear();

    const float pi = 3.14159265358979323846f;

    for (int i = 0; i <= stacks; ++i) {
        const float v = static_cast<float>(i) / static_cast<float>(stacks);
        const float phi = v * pi;
        const float y = std::cos(phi);
        const float sin_phi = std::sin(phi);

        for (int j = 0; j <= slices; ++j) {
            const float u = static_cast<float>(j) / static_cast<float>(slices);
            const float theta = u * (2.0f * pi);

            const float x = sin_phi * std::cos(theta);
            const float z = sin_phi * std::sin(theta);

            vertices.push_back(x);
            vertices.push_back(y);
            vertices.push_back(z);
            vertices.push_back(x);
            vertices.push_back(y);
            vertices.push_back(z);
        }
    }

    for (int i = 0; i < stacks; ++i) {
        for (int j = 0; j < slices; ++j) {
            const uint32_t first = static_cast<uint32_t>(i * (slices + 1) + j);
            const uint32_t second = first + static_cast<uint32_t>(slices + 1);

            indices.push_back(first);
            indices.push_back(second);
            indices.push_back(first + 1);

            indices.push_back(second);
            indices.push_back(second + 1);
            indices.push_back(first + 1);
        }
    }
}
} 

Renderer::Renderer(int width, int height)
{
    const GLuint vertex_shader = CompileShader(GL_VERTEX_SHADER, kVertexShader);
    const GLuint fragment_shader = CompileShader(GL_FRAGMENT_SHADER, kFragmentShader);
    mProgram = LinkProgram(vertex_shader, fragment_shader);

    glDeleteShader(vertex_shader);
    glDeleteShader(fragment_shader);

    glGenVertexArrays(1, &mCubeVao);
    glGenBuffers(1, &mCubeVbo);

    glBindVertexArray(mCubeVao);
    glBindBuffer(GL_ARRAY_BUFFER, mCubeVbo);
    glBufferData(GL_ARRAY_BUFFER, sizeof(kCubeVertices), kCubeVertices, GL_STATIC_DRAW);

    constexpr GLsizei stride = 6 * sizeof(float);
    glEnableVertexAttribArray(0);
    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, stride, reinterpret_cast<void*>(0));
    glEnableVertexAttribArray(1);
    glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, stride, reinterpret_cast<void*>(3 * sizeof(float)));

    glBindBuffer(GL_ARRAY_BUFFER, 0);
    glBindVertexArray(0);

    std::vector<float> sphere_vertices;
    std::vector<uint32_t> sphere_indices;
    BuildSphereMesh(20, 32, sphere_vertices, sphere_indices);
    mSphereIndexCount = static_cast<GLsizei>(sphere_indices.size());

    glGenVertexArrays(1, &mSphereVao);
    glGenBuffers(1, &mSphereVbo);
    glGenBuffers(1, &mSphereEbo);

    glBindVertexArray(mSphereVao);
    glBindBuffer(GL_ARRAY_BUFFER, mSphereVbo);
    glBufferData(GL_ARRAY_BUFFER, sphere_vertices.size() * sizeof(float), sphere_vertices.data(), GL_STATIC_DRAW);

    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, mSphereEbo);
    glBufferData(GL_ELEMENT_ARRAY_BUFFER, sphere_indices.size() * sizeof(uint32_t), sphere_indices.data(), GL_STATIC_DRAW);

    glEnableVertexAttribArray(0);
    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, stride, reinterpret_cast<void*>(0));
    glEnableVertexAttribArray(1);
    glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, stride, reinterpret_cast<void*>(3 * sizeof(float)));

    glBindVertexArray(0);

    const float aspect = height > 0 ? static_cast<float>(width) / static_cast<float>(height) : 1.0f;
    mProjection = glm::perspective(glm::radians(45.0f), aspect, 0.1f, 500.0f);

    mLights[0] = {glm::vec3(15.0f, 30.0f, 15.0f), glm::vec3(1.0f, 0.95f, 0.9f), 80.0f};
    mLights[1] = {glm::vec3(-15.0f, 25.0f, -10.0f), glm::vec3(0.6f, 0.7f, 1.0f), 50.0f};
    mLights[2] = {glm::vec3(0.0f, 15.0f, 20.0f), glm::vec3(1.0f, 1.0f, 1.0f), 40.0f};
    mLights[3] = {glm::vec3(-20.0f, 10.0f, 5.0f), glm::vec3(1.0f, 0.5f, 0.3f), 30.0f};

    mModelLoc = glGetUniformLocation(mProgram, "uModel");
    mViewLoc = glGetUniformLocation(mProgram, "uView");
    mProjLoc = glGetUniformLocation(mProgram, "uProj");
    mViewPosLoc = glGetUniformLocation(mProgram, "uViewPos");
    mObjectColorLoc = glGetUniformLocation(mProgram, "uObjectColor");
    mMetallicLoc = glGetUniformLocation(mProgram, "uMetallic");
    mRoughnessLoc = glGetUniformLocation(mProgram, "uRoughness");
    mNumLightsLoc = glGetUniformLocation(mProgram, "uNumLights");
    
    for (int i = 0; i < 4; ++i) {
        char name[32];
        snprintf(name, sizeof(name), "uLights[%d].position", i);
        mLightPosLoc[i] = glGetUniformLocation(mProgram, name);
        snprintf(name, sizeof(name), "uLights[%d].color", i);
        mLightColorLoc[i] = glGetUniformLocation(mProgram, name);
        snprintf(name, sizeof(name), "uLights[%d].intensity", i);
        mLightIntensityLoc[i] = glGetUniformLocation(mProgram, name);
    }
}

Renderer::~Renderer()
{
    if (mProgram != 0) glDeleteProgram(mProgram);
    if (mCubeVbo != 0) glDeleteBuffers(1, &mCubeVbo);
    if (mCubeVao != 0) glDeleteVertexArrays(1, &mCubeVao);
    if (mSphereEbo != 0) glDeleteBuffers(1, &mSphereEbo);
    if (mSphereVbo != 0) glDeleteBuffers(1, &mSphereVbo);
    if (mSphereVao != 0) glDeleteVertexArrays(1, &mSphereVao);
}

void Renderer::Draw(JPH::PhysicsSystem* physicsSystem, const glm::vec3& cameraPos, int envIndex)
{
    glClearColor(0.02f, 0.02f, 0.05f, 1.0f);
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

    if (physicsSystem == nullptr || mProgram == 0) return;

    mView = glm::lookAt(cameraPos, glm::vec3(0.0f, 0.0f, 0.0f), glm::vec3(0.0f, 1.0f, 0.0f));
    mViewPosition = cameraPos;

    glUseProgram(mProgram);
    glUniformMatrix4fv(mViewLoc, 1, GL_FALSE, glm::value_ptr(mView));
    glUniformMatrix4fv(mProjLoc, 1, GL_FALSE, glm::value_ptr(mProjection));
    glUniform3fv(mViewPosLoc, 1, glm::value_ptr(mViewPosition));
    glUniform1i(mNumLightsLoc, 4);
    
    for (int i = 0; i < 4; ++i) {
        glUniform3fv(mLightPosLoc[i], 1, glm::value_ptr(mLights[i].position));
        glUniform3fv(mLightColorLoc[i], 1, glm::value_ptr(mLights[i].color));
        glUniform1f(mLightIntensityLoc[i], mLights[i].intensity);
    }

    JPH::BodyInterface& body_interface = physicsSystem->GetBodyInterface();
    
    JPH::BodyIDVector bodies;
    physicsSystem->GetBodies(bodies);

    const JPH::ObjectLayer staticLayer = Layers::STATIC;
    const JPH::ObjectLayer envBaseLayer = Layers::MOVING_BASE + envIndex;

    for (const JPH::BodyID& body_id : bodies) {
        if (body_id.IsInvalid()) continue;

        JPH::ObjectLayer layer = body_interface.GetObjectLayer(body_id);
        
        if (layer != staticLayer && layer != envBaseLayer) {
            continue; 
        }

        JPH::RefConst<JPH::Shape> shape = body_interface.GetShape(body_id);
        const JPH::Shape* shape_ptr = shape.GetPtr();
        if (shape_ptr == nullptr) continue;

        glm::vec3 scale(1.0f);
        bool draw_sphere = false;

        switch (shape_ptr->GetSubType()) {
        case JPH::EShapeSubType::Sphere: {
            const auto* sphere = static_cast<const JPH::SphereShape*>(shape_ptr);
            scale = glm::vec3(sphere->GetRadius());
            draw_sphere = true;
            break;
        }
        case JPH::EShapeSubType::Box: {
            const auto* box = static_cast<const JPH::BoxShape*>(shape_ptr);
            const JPH::Vec3 half = box->GetHalfExtent();
            scale = glm::vec3(half.GetX() * 2.0f, half.GetY() * 2.0f, half.GetZ() * 2.0f);
            break;
        }
        case JPH::EShapeSubType::Cylinder: {
            const auto* cylinder = static_cast<const JPH::CylinderShape*>(shape_ptr);
            scale = glm::vec3(cylinder->GetRadius(), cylinder->GetHalfHeight() * 2.0f, cylinder->GetRadius());
            break;
        }
        default: {
            const JPH::Vec3 extent = shape_ptr->GetLocalBounds().GetExtent();
            scale = glm::vec3(extent.GetX() * 2.0f, extent.GetY() * 2.0f, extent.GetZ() * 2.0f);
            break;
        }
        }

        const JPH::RMat44 transform = body_interface.GetWorldTransform(body_id);
        glm::mat4 model = ToGlmMat4(transform);
        model = model * glm::scale(glm::mat4(1.0f), scale);

        glUniformMatrix4fv(mModelLoc, 1, GL_FALSE, glm::value_ptr(model));
        
        const int body_index = static_cast<int>(body_id.GetIndex());
        glm::vec3 objectColor;
        float metallic = 0.9f;
        float roughness = 0.1f;
        
        if (layer == staticLayer) {
            objectColor = glm::vec3(0.4f, 0.4f, 0.4f);
            metallic = 0.1f;
            roughness = 0.9f;
        } else if (body_index % 3 == 0) {
            objectColor = glm::vec3(0.9f, 0.2f, 0.1f);
            metallic = 0.8f;
            roughness = 0.2f;
        } else if (body_index % 3 == 1) {
            objectColor = glm::vec3(0.2f, 0.4f, 0.9f);
            metallic = 0.95f;
            roughness = 0.05f;
        } else {
            objectColor = glm::vec3(1.0f, 0.9f, 0.1f);
            metallic = 1.0f;
            roughness = 0.02f;
        }
        
        glUniform3fv(mObjectColorLoc, 1, glm::value_ptr(objectColor));
        glUniform1f(mMetallicLoc, metallic);
        glUniform1f(mRoughnessLoc, roughness);

        if (draw_sphere) {
            glBindVertexArray(mSphereVao);
            glDrawElements(GL_TRIANGLES, mSphereIndexCount, GL_UNSIGNED_INT, nullptr);
        } else {
            glBindVertexArray(mCubeVao);
            glDrawArrays(GL_TRIANGLES, 0, 36);
        }
    }

    glBindVertexArray(0);
    glUseProgram(0);
}

GLuint Renderer::CompileShader(GLenum type, const char* source)
{
    GLuint shader = glCreateShader(type);
    glShaderSource(shader, 1, &source, nullptr);
    glCompileShader(shader);
    return shader;
}

GLuint Renderer::LinkProgram(GLuint vertexShader, GLuint fragmentShader)
{
    GLuint program = glCreateProgram();
    glAttachShader(program, vertexShader);
    glAttachShader(program, fragmentShader);
    glLinkProgram(program);
    return program;
}

glm::mat4 Renderer::ToGlmMat4(const JPH::RMat44& mat) const
{
    glm::mat4 out(1.0f);
    for (int c = 0; c < 4; ++c) {
        for (int r = 0; r < 4; ++r) {
            out[c][r] = static_cast<float>(mat(static_cast<uint>(r), static_cast<uint>(c)));
        }
    }
    return out;
}