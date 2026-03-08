// STRICT REQUIREMENT: Jolt.h must be included first
#include <Jolt/Jolt.h>
#include "Renderer.h"
#include "PhysicsCore.h" // We need this for the Dimensional Layers

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <iostream>
#include <limits>
#include <vector>

#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtc/type_ptr.hpp>

#include <Jolt/Physics/Body/BodyInterface.h>
#include <Jolt/Physics/Collision/Shape/BoxShape.h>
#include <Jolt/Physics/Collision/Shape/Shape.h>
#include <Jolt/Physics/Collision/Shape/SphereShape.h>
#include <Jolt/Physics/Collision/Shape/CylinderShape.h>
#include <Jolt/Physics/Collision/Shape/StaticCompoundShape.h>
#include <Jolt/Physics/Collision/Shape/MutableCompoundShape.h>

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

out vec3 Normal;

void main()
{
    gl_Position = uProj * uView * uModel * vec4(aPos, 1.0);
    Normal = mat3(transpose(inverse(uModel))) * aNormal;
}
)";

constexpr const char* kFragmentShader = R"(
#version 330 core
out vec4 FragColor;

in vec3 Normal;

uniform vec3 uObjectColor;
uniform float uAlpha;

void main()
{
    // Dead simple: Light from top-front, two-sided
    vec3 n = normalize(Normal);
    vec3 lightDir = normalize(vec3(0.3, 1.0, 0.5));
    float diff = abs(dot(n, lightDir)) * 0.6 + 0.4;
    FragColor = vec4(uObjectColor * diff, uAlpha);
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
    mProjection = glm::perspective(glm::radians(45.0f), aspect, 0.1f, 10000.0f);

    mLights[0] = {glm::vec3(15.0f, 30.0f, 15.0f), glm::vec3(1.0f, 0.95f, 0.9f), 80.0f};
    mLights[1] = {glm::vec3(-15.0f, 25.0f, -10.0f), glm::vec3(0.6f, 0.7f, 1.0f), 50.0f};
    mLights[2] = {glm::vec3(0.0f, 15.0f, 20.0f), glm::vec3(1.0f, 1.0f, 1.0f), 40.0f};
    mLights[3] = {glm::vec3(-20.0f, 10.0f, 5.0f), glm::vec3(1.0f, 0.5f, 0.3f), 30.0f};

    mModelLoc = glGetUniformLocation(mProgram, "uModel");
    mViewLoc = glGetUniformLocation(mProgram, "uView");
    mProjLoc = glGetUniformLocation(mProgram, "uProj");
    mViewPosLoc = glGetUniformLocation(mProgram, "uViewPos");
    mObjectColorLoc = glGetUniformLocation(mProgram, "uObjectColor");
    mAlphaLoc = glGetUniformLocation(mProgram, "uAlpha");
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

void Renderer::Resize(int width, int height) {
    const float aspect = height > 0 ? static_cast<float>(width) / static_cast<float>(height) : 1.0f;
    mProjection = glm::perspective(glm::radians(45.0f), aspect, 0.1f, 10000.0f);
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

void Renderer::Draw(PhysicsCore* physicsCore, const glm::vec3& cameraPos, int envIndex, const glm::vec3& cameraFront, const glm::vec3& cameraUp,
                    bool showCollisionShapes, bool showAABBs, bool showAABBs_unused, bool showRobot1, bool showRobot2,
                    const EnvVisualState* visualState)
{
    // ... setup code ...
    glEnable(GL_DEPTH_TEST);
    glDepthFunc(GL_LESS);
    glDisable(GL_CULL_FACE);
    glClearColor(0.15f, 0.15f, 0.2f, 1.0f); 
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

    if (physicsCore == nullptr || mProgram == 0) return;

    JPH::PhysicsSystem* physicsSystem = &physicsCore->GetPhysicsSystem();
    JPH::BodyInterface& body_interface = physicsSystem->GetBodyInterface();

    mView = glm::lookAt(cameraPos, cameraPos + cameraFront, cameraUp);
    mViewPosition = cameraPos;

    glUseProgram(mProgram);
    glUniformMatrix4fv(mViewLoc, 1, GL_FALSE, glm::value_ptr(mView));
    glUniformMatrix4fv(mProjLoc, 1, GL_FALSE, glm::value_ptr(mProjection));
    glUniform3fv(mViewPosLoc, 1, glm::value_ptr(mViewPosition));
    glUniform1i(mNumLightsLoc, 4);
    
    for (int i = 0; i < 4; ++i) {
        glm::vec3 lightPos = cameraPos + mLights[i].position;
        glUniform3fv(mLightPosLoc[i], 1, glm::value_ptr(lightPos));
        glUniform3fv(mLightColorLoc[i], 1, glm::value_ptr(mLights[i].color));
        glUniform1f(mLightIntensityLoc[i], mLights[i].intensity * 100.0f);
    }

    // 1. RENDER STATIC ARENA (using PhysicsCore)
    const JPH::ObjectLayer staticLayer = Layers::STATIC;
    JPH::BodyIDVector staticBodies;
    physicsCore->GetBodiesByLayers(staticBodies, { staticLayer });

    auto renderRaw = [&](const JPH::Shape* s, const JPH::RMat44& transform, const glm::vec3& color, float alpha) {
        if (!s) return;
        glm::vec3 scale(1.0f);
        bool is_sphere = false;
        if (s->GetSubType() == JPH::EShapeSubType::Sphere) {
            scale = glm::vec3(static_cast<const JPH::SphereShape*>(s)->GetRadius());
            is_sphere = true;
        } else if (s->GetSubType() == JPH::EShapeSubType::Box) {
            const JPH::Vec3 half = static_cast<const JPH::BoxShape*>(s)->GetHalfExtent();
            scale = glm::vec3(half.GetX() * 2.0f, half.GetY() * 2.0f, half.GetZ() * 2.0f);
        }
        glm::mat4 model = ToGlmMat4(transform) * glm::scale(glm::mat4(1.0f), scale);
        glUniformMatrix4fv(mModelLoc, 1, GL_FALSE, glm::value_ptr(model));
        glUniform3fv(mObjectColorLoc, 1, glm::value_ptr(color));
        glUniform1f(mAlphaLoc, alpha);
        if (is_sphere) { glBindVertexArray(mSphereVao); glDrawElements(GL_TRIANGLES, mSphereIndexCount, GL_UNSIGNED_INT, nullptr); }
        else { glBindVertexArray(mCubeVao); glDrawArrays(GL_TRIANGLES, 0, 36); }
    };

    for (const auto& bid : staticBodies) {
        renderRaw(body_interface.GetShape(bid).GetPtr(), body_interface.GetWorldTransform(bid), glm::vec3(0.4f, 0.4f, 0.4f), 1.0f);
    }

    // 2. RENDER ROBOTS (From visualState buffer if available, else live)
    if (visualState) {
        if (showRobot1) {
            JPH::RMat44 t1 = JPH::RMat44::sRotationTranslation(
                JPH::Quat(visualState->r1.rx, visualState->r1.ry, visualState->r1.rz, visualState->r1.rw), 
                JPH::Vec3(visualState->r1.x, visualState->r1.y, visualState->r1.z)
            );
            // Use sphere shape for robot 1 visual (approximate for SPS)
            JPH::SphereShape s1(0.5f);
            renderRaw(&s1, t1, glm::vec3(0.0f, 0.8f, 0.8f), 1.0f);
        }
        if (showRobot2) {
            JPH::RMat44 t2 = JPH::RMat44::sRotationTranslation(
                JPH::Quat(visualState->r2.rx, visualState->r2.ry, visualState->r2.rz, visualState->r2.rw), 
                JPH::Vec3(visualState->r2.x, visualState->r2.y, visualState->r2.z)
            );
            // Use sphere shape for robot 2 visual (approximate for SPS)
            JPH::SphereShape s2(0.5f);
            renderRaw(&s2, t2, glm::vec3(0.8f, 0.0f, 0.8f), 1.0f);
        }
    } else {
        // Fallback: Query live Jolt bodies (Slow, blocks Sim)
        const JPH::ObjectLayer envBaseLayer = Layers::MOVING_BASE + envIndex;
        const JPH::ObjectLayer movingBaseLayer = Layers::MOVING_BASE;
        const JPH::ObjectLayer ghostLayer = Layers::GHOST_BASE + envIndex;
        
        JPH::BodyIDVector liveBodies;
        physicsCore->GetBodiesByLayers(liveBodies, { envBaseLayer, movingBaseLayer, ghostLayer });
        for (const auto& bid : liveBodies) {
            renderRaw(body_interface.GetShape(bid).GetPtr(), body_interface.GetWorldTransform(bid), glm::vec3(1.0f, 1.0f, 0.0f), 1.0f);
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

    GLint success;
    glGetShaderiv(shader, GL_COMPILE_STATUS, &success);
    if (!success) {
        char infoLog[512];
        glGetShaderInfoLog(shader, 512, nullptr, infoLog);
        std::cerr << "Shader Compilation Error (" << (type == GL_VERTEX_SHADER ? "Vertex" : "Fragment") << "):\n" << infoLog << std::endl;
    }
    return shader;
}

GLuint Renderer::LinkProgram(GLuint vertexShader, GLuint fragmentShader)
{
    GLuint program = glCreateProgram();
    glAttachShader(program, vertexShader);
    glAttachShader(program, fragmentShader);
    glLinkProgram(program);

    GLint success;
    glGetProgramiv(program, GL_LINK_STATUS, &success);
    if (!success) {
        char infoLog[512];
        glGetProgramInfoLog(program, 512, nullptr, infoLog);
        std::cerr << "Shader Linking Error:\n" << infoLog << std::endl;
    }
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