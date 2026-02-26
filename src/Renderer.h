#pragma once

// STRICT REQUIREMENT: Jolt.h must be included first
#include <Jolt/Jolt.h>
#include <vector>

#include <GL/glew.h>
#include <glm/glm.hpp>
#include <Jolt/Physics/Body/BodyID.h>
#include <Jolt/Physics/PhysicsSystem.h>

struct Light {
    glm::vec3 position;
    glm::vec3 color;
    float intensity;
};

struct Camera {
    float distance = 15.0f;
    float pitch = 0.4f;
    float yaw = 0.0f;
    glm::vec3 target = {0.0f, 0.0f, 0.0f};
};

class Renderer
{
public:
    Renderer(int width, int height);
    ~Renderer();

    void Draw(JPH::PhysicsSystem* physicsSystem, const glm::vec3& cameraPos, int envIndex = 0, const glm::vec3& cameraFront = glm::vec3(0.0f, 0.0f, -1.0f));

private:
    GLuint CompileShader(GLenum type, const char* source);
    GLuint LinkProgram(GLuint vertexShader, GLuint fragmentShader);
    glm::mat4 ToGlmMat4(const JPH::RMat44& mat) const;

    GLuint mProgram = 0;

    GLuint mCubeVao = 0;
    GLuint mCubeVbo = 0;

    GLuint mSphereVao = 0;
    GLuint mSphereVbo = 0;
    GLuint mSphereEbo = 0;
    GLsizei mSphereIndexCount = 0;

    GLint mAlphaLoc = -1;
    GLint mModelLoc = -1;
    GLint mViewLoc = -1;
    GLint mProjLoc = -1;
    GLint mViewPosLoc = -1;
    GLint mObjectColorLoc = -1;
    GLint mMetallicLoc = -1;
    GLint mRoughnessLoc = -1;
    GLint mNumLightsLoc = -1;
    GLint mLightPosLoc[4] = {-1};
    GLint mLightColorLoc[4] = {-1};
    GLint mLightIntensityLoc[4] = {-1};

    glm::mat4 mView{1.0f};
    glm::mat4 mProjection{1.0f};
    glm::vec3 mViewPosition{0.0f, 20.0f, 50.0f};
    Light mLights[4];
};