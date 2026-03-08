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
                    bool showCollisionShapes, bool showAABBs, bool showContactPoints, bool showRobot1, bool showRobot2)
{
    // TEMPORARILY DISABLE DEBUG DRAWING - causes crashes
    showCollisionShapes = false;
    showAABBs = false;
    
    glEnable(GL_DEPTH_TEST);
    glDepthFunc(GL_LESS);
    glDisable(GL_CULL_FACE); // Turn off culling

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
        glm::vec3 lightPos = cameraPos + mLights[i].position; // Relative to camera
        glUniform3fv(mLightPosLoc[i], 1, glm::value_ptr(lightPos));
        glUniform3fv(mLightColorLoc[i], 1, glm::value_ptr(mLights[i].color));
        glUniform1f(mLightIntensityLoc[i], mLights[i].intensity * 100.0f); // Increase intensity for large scale
    }

    const JPH::ObjectLayer staticLayer = Layers::STATIC;
    const JPH::ObjectLayer envBaseLayer = Layers::MOVING_BASE + envIndex;
    const JPH::ObjectLayer movingBaseLayer = Layers::MOVING_BASE;
    const JPH::ObjectLayer ghostLayer = Layers::GHOST_BASE + envIndex;
    const std::vector<JPH::ObjectLayer> relevantLayers = { staticLayer, envBaseLayer, movingBaseLayer, ghostLayer };

    JPH::BodyIDVector bodies;
    physicsCore->GetBodiesByLayers(bodies, relevantLayers);


    auto renderBody = [&](const JPH::BodyID& body_id, float forcedAlpha = -1.0f) {
        JPH::ObjectLayer layer = body_interface.GetObjectLayer(body_id);
        JPH::RefConst<JPH::Shape> shape = body_interface.GetShape(body_id);
        const JPH::Shape* shape_ptr = shape.GetPtr();
        if (shape_ptr == nullptr) return;

        const JPH::RMat44 worldTransform = body_interface.GetWorldTransform(body_id);
        
        auto drawShape = [this, &body_interface, &body_id, &layer, &staticLayer, &ghostLayer, forcedAlpha](const JPH::Shape* s, const JPH::RMat44& transform, auto& self) -> void {
            if (s->GetSubType() == JPH::EShapeSubType::StaticCompound || s->GetSubType() == JPH::EShapeSubType::MutableCompound) {
                const auto* compound = static_cast<const JPH::StaticCompoundShape*>(s);
                for (uint32_t i = 0; i < compound->GetNumSubShapes(); ++i) {
                    const auto& sub = compound->GetSubShape(i);
                    JPH::RMat44 subTransform = JPH::RMat44::sRotationTranslation(sub.GetRotation(), JPH::Vec3(sub.mPositionCOM));
                    self(sub.mShape, transform * subTransform, self);
                }
                return;
            }

            glm::vec3 scale(1.0f);
            bool draw_sphere = false;

            switch (s->GetSubType()) {
            case JPH::EShapeSubType::Sphere: {
                const auto* sphere = static_cast<const JPH::SphereShape*>(s);
                scale = glm::vec3(sphere->GetRadius());
                draw_sphere = true;
                break;
            }
            case JPH::EShapeSubType::Box: {
                const auto* box = static_cast<const JPH::BoxShape*>(s);
                const JPH::Vec3 half = box->GetHalfExtent();
                scale = glm::vec3(half.GetX() * 2.0f, half.GetY() * 2.0f, half.GetZ() * 2.0f);
                break;
            }
            case JPH::EShapeSubType::Cylinder: {
                const auto* cylinder = static_cast<const JPH::CylinderShape*>(s);
                scale = glm::vec3(cylinder->GetRadius(), cylinder->GetHalfHeight() * 2.0f, cylinder->GetRadius());
                break;
            }
            default: {
                const JPH::Vec3 extent = s->GetLocalBounds().GetExtent();
                scale = glm::vec3(extent.GetX() * 2.0f, extent.GetY() * 2.0f, extent.GetZ() * 2.0f);
                break;
            }
            }

            glm::mat4 model = ToGlmMat4(transform);
            model = model * glm::scale(glm::mat4(1.0f), scale);

            glUniformMatrix4fv(mModelLoc, 1, GL_FALSE, glm::value_ptr(model));
            
            const int body_index = static_cast<int>(body_id.GetIndex());
            glm::vec3 objectColor;
            float metallic = 0.9f;
            float roughness = 0.1f;
            float alpha = (forcedAlpha > 0.0f) ? forcedAlpha : 1.0f;
            
            if (layer == staticLayer) {
                objectColor = glm::vec3(0.4f, 0.4f, 0.4f);
                metallic = 0.1f;
                roughness = 0.9f;
                if (transform.GetTranslation().GetY() < -0.1f) objectColor = glm::vec3(0.2f, 0.2f, 0.25f);
            } else if (layer == ghostLayer) {
                objectColor = glm::vec3(1.0f, 0.2f, 0.2f);
                alpha = 0.6f;
                metallic = 0.5f;
                roughness = 0.5f;
            } else if (body_index % 3 == 0) {
                objectColor = glm::vec3(0.0f, 0.8f, 0.8f);
            } else if (body_index % 3 == 1) {
                objectColor = glm::vec3(0.8f, 0.0f, 0.8f);
            } else {
                objectColor = glm::vec3(1.0f, 0.9f, 0.1f);
            }
            
            glUniform3fv(mObjectColorLoc, 1, glm::value_ptr(objectColor));
            glUniform1f(mMetallicLoc, metallic);
            glUniform1f(mRoughnessLoc, roughness);
            glUniform1f(mAlphaLoc, alpha);

            if (draw_sphere) {
                glBindVertexArray(mSphereVao);
                glDrawElements(GL_TRIANGLES, mSphereIndexCount, GL_UNSIGNED_INT, nullptr);
            } else {
                glBindVertexArray(mCubeVao);
                glDrawArrays(GL_TRIANGLES, 0, 36);
            }
        };

        drawShape(shape_ptr, worldTransform, drawShape);
    };

    // Pass 1: Opaque
    for (const JPH::BodyID& body_id : bodies) {
        if (body_id.IsInvalid()) continue;
        JPH::ObjectLayer layer = body_interface.GetObjectLayer(body_id);
        JPH::RVec3 pos = body_interface.GetCenterOfMassPosition(body_id);
        
        bool isWall = (layer == staticLayer && pos.GetY() > 1.0f);
        if (!isWall) {
            renderBody(body_id, 1.0f);
        }
    }

    // Pass 2: Transparent Walls
    glEnable(GL_BLEND);
    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
    glDepthMask(GL_FALSE);

    for (const JPH::BodyID& body_id : bodies) {
        if (body_id.IsInvalid()) continue;
        JPH::ObjectLayer layer = body_interface.GetObjectLayer(body_id);
        JPH::RVec3 pos = body_interface.GetCenterOfMassPosition(body_id);
        
        bool isWall = (layer == staticLayer && pos.GetY() > 1.0f);
        if (isWall) {
            renderBody(body_id, 0.3f);
        }
    }

    glDepthMask(GL_TRUE);
    glDisable(GL_BLEND);

    // Debug drawing: wireframe collision shapes
    if (showCollisionShapes) {
        glPolygonMode(GL_FRONT_AND_BACK, GL_LINE);
        
        for (const JPH::BodyID& body_id : bodies) {
            if (body_id.IsInvalid()) continue;
            JPH::ObjectLayer layer = body_interface.GetObjectLayer(body_id);
            if (layer != staticLayer && layer != envBaseLayer && layer != ghostLayer) continue;

            JPH::RefConst<JPH::Shape> shape = body_interface.GetShape(body_id);
            const JPH::Shape* shape_ptr = shape.GetPtr();
            if (shape_ptr == nullptr) continue;

            const JPH::RMat44 worldTransform = body_interface.GetWorldTransform(body_id);

            auto drawWireframeShape = [&](const JPH::Shape* s, const JPH::RMat44& transform, auto& self) -> void {
                if (s->GetSubType() == JPH::EShapeSubType::StaticCompound || s->GetSubType() == JPH::EShapeSubType::MutableCompound) {
                    const auto* compound = static_cast<const JPH::StaticCompoundShape*>(s);
                    for (uint32_t i = 0; i < compound->GetNumSubShapes(); ++i) {
                        const auto& sub = compound->GetSubShape(i);
                        JPH::RMat44 subTransform = JPH::RMat44::sRotationTranslation(sub.GetRotation(), JPH::Vec3(sub.mPositionCOM));
                        self(sub.mShape, transform * subTransform, self);
                    }
                    return;
                }

                glm::vec3 scale(1.0f);
                bool draw_sphere = false;

                switch (s->GetSubType()) {
                case JPH::EShapeSubType::Sphere: {
                    const auto* sphere = static_cast<const JPH::SphereShape*>(s);
                    scale = glm::vec3(sphere->GetRadius());
                    draw_sphere = true;
                    break;
                }
                case JPH::EShapeSubType::Box: {
                    const auto* box = static_cast<const JPH::BoxShape*>(s);
                    const JPH::Vec3 half = box->GetHalfExtent();
                    scale = glm::vec3(half.GetX() * 2.0f, half.GetY() * 2.0f, half.GetZ() * 2.0f);
                    break;
                }
                default: {
                    // Fallback for Cylinders, Convex Hulls, etc.
                    JPH::Shape::GetTrianglesContext ctx;

                    s->GetTrianglesStart(ctx, JPH::AABox::sBiggest(), JPH::Vec3::sZero(), JPH::Quat::sIdentity(), JPH::Vec3::sReplicate(1.0f));
                    JPH::Float3* vertices = new JPH::Float3[4096]; // Buffer
                    int count = s->GetTrianglesNext(ctx, 4096, vertices, nullptr);
                    
                    if (count > 0) {
                        glm::mat4 model = ToGlmMat4(transform);
                        glUniformMatrix4fv(mModelLoc, 1, GL_FALSE, glm::value_ptr(model));
                        
                        // Setup material props (same as boxes)
                        const int body_index = static_cast<int>(body_id.GetIndex());
                        glm::vec3 objectColor;
                        float metallic = 0.9f;
                        float roughness = 0.1f;
                        float alpha = 1.0f;  // Default alpha for fallback rendering
                        
                        if (layer == staticLayer) {
                            objectColor = glm::vec3(0.4f, 0.4f, 0.4f);
                            metallic = 0.1f;
                            roughness = 0.9f;
                        } else if (layer == ghostLayer) {
                            objectColor = glm::vec3(1.0f, 0.2f, 0.2f);
                            alpha = 0.6f;
                            metallic = 0.5f;
                            roughness = 0.5f;
                        } else if (body_index % 3 == 0) {
                            objectColor = glm::vec3(0.0f, 0.8f, 0.8f);
                        } else if (body_index % 3 == 1) {
                            objectColor = glm::vec3(0.8f, 0.0f, 0.8f);
                        } else {
                            objectColor = glm::vec3(1.0f, 0.9f, 0.1f);
                        }
                        
                        glUniform3fv(mObjectColorLoc, 1, glm::value_ptr(objectColor));
                        glUniform1f(mMetallicLoc, metallic);
                        glUniform1f(mRoughnessLoc, roughness);
                        glUniform1f(mAlphaLoc, alpha);

                        // Immediate mode style drawing using a dynamic VAO would be better, 
                        // but for now let's just use a temporary buffer and draw.
                        // Ideally we should cache this VAO in the shape UserData.
                        
                        std::vector<float> triVerts;
                        triVerts.reserve(count * 3 * 6); // Pos + Normal
                        
                        for (int i = 0; i < count; ++i) {
                            JPH::Vec3 v1(vertices[i*3+0].x, vertices[i*3+0].y, vertices[i*3+0].z);
                            JPH::Vec3 v2(vertices[i*3+1].x, vertices[i*3+1].y, vertices[i*3+1].z);
                            JPH::Vec3 v3(vertices[i*3+2].x, vertices[i*3+2].y, vertices[i*3+2].z);
                            JPH::Vec3 normal = (v2 - v1).Cross(v3 - v1).Normalized();
                            
                            auto push = [&](const JPH::Vec3& v) {
                                triVerts.push_back(v.GetX()); triVerts.push_back(v.GetY()); triVerts.push_back(v.GetZ());
                                triVerts.push_back(normal.GetX()); triVerts.push_back(normal.GetY()); triVerts.push_back(normal.GetZ());
                            };
                            push(v1); push(v2); push(v3);
                        }

                        // Use the Cube VAO as a scratch buffer if we update it? No, unsafe.
                        // Let's create a temporary VAO/VBO for this draw call (Slow but works for this viewer)
                        GLuint vao, vbo;
                        glGenVertexArrays(1, &vao);
                        glGenBuffers(1, &vbo);
                        
                        glBindVertexArray(vao);
                        glBindBuffer(GL_ARRAY_BUFFER, vbo);
                        glBufferData(GL_ARRAY_BUFFER, triVerts.size() * sizeof(float), triVerts.data(), GL_STREAM_DRAW);
                        
                        glEnableVertexAttribArray(0);
                        glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 6 * sizeof(float), (void*)0);
                        glEnableVertexAttribArray(1);
                        glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, 6 * sizeof(float), (void*)(3 * sizeof(float)));
                        
                        glDrawArrays(GL_TRIANGLES, 0, count * 3);
                        
                        glDeleteBuffers(1, &vbo);
                        glDeleteVertexArrays(1, &vao);
                    }
                    
                    delete[] vertices;
                    return; // Done
                }
                }

                glm::mat4 model = ToGlmMat4(transform);
                model = model * glm::scale(glm::mat4(1.0f), scale);

                glUniformMatrix4fv(mModelLoc, 1, GL_FALSE, glm::value_ptr(model));
                glUniform3f(mObjectColorLoc, 1.0f, 1.0f, 0.0f);
                glUniform1f(mMetallicLoc, 0.0f);
                glUniform1f(mRoughnessLoc, 1.0f);
                glUniform1f(mAlphaLoc, 1.0f);

                if (draw_sphere) {
                    glBindVertexArray(mSphereVao);
                    glDrawElements(GL_TRIANGLES, mSphereIndexCount, GL_UNSIGNED_INT, nullptr);
                } else {
                    glBindVertexArray(mCubeVao);
                    glDrawArrays(GL_TRIANGLES, 0, 36);
                }
            };

            drawWireframeShape(shape_ptr, worldTransform, drawWireframeShape);
        }

        glPolygonMode(GL_FRONT_AND_BACK, GL_FILL);
    }

    // Debug drawing: AABBs
    if (showAABBs) {
        glPolygonMode(GL_FRONT_AND_BACK, GL_LINE);

        for (const JPH::BodyID& body_id : bodies) {
            if (body_id.IsInvalid()) continue;
            JPH::ObjectLayer layer = body_interface.GetObjectLayer(body_id);
            if (layer != staticLayer && layer != envBaseLayer && layer != ghostLayer) continue;

            JPH::RefConst<JPH::Shape> shape = body_interface.GetShape(body_id);
            const JPH::Shape* shape_ptr = shape.GetPtr();
            if (shape_ptr == nullptr) continue;

            JPH::AABox localBounds = shape_ptr->GetLocalBounds();
            const JPH::Vec3& min = localBounds.mMin;
            const JPH::Vec3& max = localBounds.mMax;

            JPH::RMat44 worldTransform = body_interface.GetWorldTransform(body_id);

            JPH::Vec3 worldMin(std::numeric_limits<float>::max(), std::numeric_limits<float>::max(), std::numeric_limits<float>::max());
            JPH::Vec3 worldMax(std::numeric_limits<float>::lowest(), std::numeric_limits<float>::lowest(), std::numeric_limits<float>::lowest());

            JPH::RVec3 corners[8] = {
                worldTransform * JPH::RVec3(min.GetX(), min.GetY(), min.GetZ()),
                worldTransform * JPH::RVec3(max.GetX(), min.GetY(), min.GetZ()),
                worldTransform * JPH::RVec3(min.GetX(), max.GetY(), min.GetZ()),
                worldTransform * JPH::RVec3(max.GetX(), max.GetY(), min.GetZ()),
                worldTransform * JPH::RVec3(min.GetX(), min.GetY(), max.GetZ()),
                worldTransform * JPH::RVec3(max.GetX(), min.GetY(), max.GetZ()),
                worldTransform * JPH::RVec3(min.GetX(), max.GetY(), max.GetZ()),
                worldTransform * JPH::RVec3(max.GetX(), max.GetY(), max.GetZ()),
            };

            for (int i = 0; i < 8; ++i) {
                worldMin = JPH::Vec3::sMin(worldMin, corners[i]);
                worldMax = JPH::Vec3::sMax(worldMax, corners[i]);
            }

            JPH::RMat44 aabbTransform = JPH::RMat44::sTranslation((worldMin + worldMax) * 0.5f);
            glm::vec3 aabbScale(max.GetX() - min.GetX(), max.GetY() - min.GetY(), max.GetZ() - min.GetZ());

            glm::mat4 model = ToGlmMat4(aabbTransform);
            model = model * glm::scale(glm::mat4(1.0f), aabbScale);

            glUniformMatrix4fv(mModelLoc, 1, GL_FALSE, glm::value_ptr(model));
            glUniform3f(mObjectColorLoc, 0.0f, 1.0f, 0.0f);
            glUniform1f(mMetallicLoc, 0.0f);
            glUniform1f(mRoughnessLoc, 1.0f);
            glUniform1f(mAlphaLoc, 1.0f);

            glBindVertexArray(mCubeVao);
            glDrawArrays(GL_TRIANGLES, 0, 36);
        }

        glPolygonMode(GL_FRONT_AND_BACK, GL_FILL);
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