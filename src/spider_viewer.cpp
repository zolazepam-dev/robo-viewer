// Simple Spider Robot Viewer
#include <GL/glew.h>
#include <GLFW/glfw3.h>
#include <GL/glu.h>
#include "imgui.h"
#include "imgui_impl_glfw.h"
#include "imgui_impl_opengl3.h"

#include <Jolt/Jolt.h>
#include <Jolt/RegisterTypes.h>
#include <Jolt/Core/Factory.h>
#include <Jolt/Core/TempAllocator.h>
#include <Jolt/Core/JobSystemThreadPool.h>
#include <Jolt/Physics/PhysicsSystem.h>
#include <Jolt/Physics/Body/BodyInterface.h>
#include <Jolt/Physics/Body/BodyCreationSettings.h>
#include <Jolt/Physics/Collision/Shape/BoxShape.h>
#include <Jolt/Physics/Collision/Shape/CapsuleShape.h>
#include <Jolt/Physics/Collision/Shape/SphereShape.h>
#include <Jolt/Physics/Constraints/HingeConstraint.h>
#include <Jolt/Physics/Constraints/MotorSettings.h>
#include <Jolt/Physics/Collision/BroadPhase/BroadPhaseLayer.h>
#include <Jolt/Physics/Collision/ObjectLayer.h>
#include <Jolt/Physics/Collision/BroadPhase/BroadPhaseLayerInterfaceTable.h>
#include <Jolt/Physics/Collision/ObjectLayerPairFilterTable.h>

#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtc/type_ptr.hpp>

#include <iostream>
#include <vector>
#include <cmath>
#include <thread>

using namespace JPH;

// ============================================================================
// Layers Configuration
// ============================================================================
namespace Layers {
    static constexpr ObjectLayer STATIC = 0;
    static constexpr ObjectLayer DYNAMIC = 1;
    static constexpr uint32_t NUM_LAYERS = 2;
    static constexpr uint32_t NUM_BROAD_PHASE_LAYERS = 2;
}

// ============================================================================
// Globals
// ============================================================================
PhysicsSystem* gPhysics = nullptr;
BodyInterface* gBodyInterface = nullptr;
TempAllocatorImpl* gTempAllocator = nullptr;
JobSystemThreadPool* gJobSystem = nullptr;

// Spider structures
struct SpiderBody {
    BodyID body;
    std::string name;
    glm::vec3 color;
};

struct SpiderJoint {
    HingeConstraint* constraint;
    std::string name;
    float targetAngle;
    float minAngle;
    float maxAngle;
};

std::vector<SpiderBody> gSpiderBodies;
std::vector<SpiderJoint> gSpiderJoints;

// Camera
float gCameraYaw = 0.0f;
float gCameraPitch = 0.3f;
float gCameraDistance = 8.0f;
bool gMoveCamera = false;
double gLastMouseX = 0, gLastMouseY = 0;

// Time
float gTime = 0.0f;

// Leg positions for 6-legged spider (hexapod)
struct LegPosition {
    float x, y;
    float coxaAngle;
};

constexpr int NUM_LEGS = 6;
LegPosition legPositions[NUM_LEGS] = {
    { 0.4f,  0.4f,  0.0f },  // Leg 0 (front right)
    { 0.0f,  0.5f,  0.0f },  // Leg 1 (right middle)
    {-0.4f,  0.4f,  0.0f },  // Leg 2 (rear right)
    {-0.4f, -0.4f,  0.0f },  // Leg 3 (rear left)
    { 0.0f, -0.5f,  0.0f },  // Leg 4 (left middle)
    { 0.4f, -0.4f,  0.0f }   // Leg 5 (front left)
};

// Shape dimensions
constexpr float CENTRAL_BOX_HALF_X = 0.3f;
constexpr float CENTRAL_BOX_HALF_Y = 0.2f;
constexpr float CENTRAL_BOX_HALF_Z = 0.15f;

constexpr float COXIA_HALF_HEIGHT = 0.08f;
constexpr float COXIA_RADIUS = 0.06f;
constexpr float COXIA_MASS = 8.0f;

constexpr float FEMUR_HALF_HEIGHT = 0.15f;
constexpr float FEMUR_RADIUS = 0.05f;
constexpr float FEMUR_MASS = 12.0f;

constexpr float TIBIA_HALF_HEIGHT = 0.18f;
constexpr float TIBIA_RADIUS = 0.04f;
constexpr float TIBIA_MASS = 10.0f;

constexpr float CENTRAL_MASS = 80.0f;

// ============================================================================
// Initialization
// ============================================================================
void InitJolt() {
    std::cout << "Initializing Jolt..." << std::endl;
    
    JPH::RegisterDefaultAllocator();
    JPH::Factory::sInstance = new JPH::Factory();
    JPH::RegisterTypes();
    
    gTempAllocator = new TempAllocatorImpl(10 * 1024 * 1024);
    int numThreads = std::max(1, (int)std::thread::hardware_concurrency() - 1);
    gJobSystem = new JobSystemThreadPool(2 * 1024 * 1024, numThreads);
    
    // Create filters
    ObjectLayerPairFilterTable* layerFilter = new ObjectLayerPairFilterTable();
    BroadPhaseLayerInterfaceTable* broadPhaseLayerInterface = new BroadPhaseLayerInterfaceTable(
        Layers::NUM_LAYERS, Layers::NUM_BROAD_PHASE_LAYERS);
    
    PhysicsSettings settings;
    settings.mStepSize = 1.0f / 60.0f;
    settings.mAllowSleeping = false;
    settings.mDeterministicSimulation = true;
    
    gPhysics = new PhysicsSystem();
    gPhysics->Init(200, 4096, *gTempAllocator, *gJobSystem, broadPhaseLayerInterface, layerFilter, settings);
    gPhysics->SetGravity(Vec3(0, -9.81f, 0));
    
    gBodyInterface = &gPhysics->GetBodyInterface();
    
    std::cout << "Jolt initialized!" << std::endl;
}

// ============================================================================
// Arena
// ============================================================================
void CreateArena() {
    std::cout << "Creating arena..." << std::endl;
    
    // Ground
    auto groundShape = BoxShapeSettings(Vec3(20.0f, 0.5f, 20.0f));
    auto groundResult = groundShape.Create();
    BodyCreationSettings groundSettings(groundResult.Get(), RVec3(0, -0.5f, 0), Quat::sIdentity(),
                                         EMotionType::Static, Layers::STATIC);
    groundSettings.mFriction = 0.8f;
    gBodyInterface->CreateAndAddBody(groundSettings, EActivation::DontActivate);
    
    std::cout << "Arena created!" << std::endl;
}

// ============================================================================
// Body Creation
// ============================================================================
BodyID CreateBody(const std::string& name, const glm::vec3& position, const glm::quat& rotation,
                  const Shape* shape, float mass, const glm::vec3& color) {
    BodyCreationSettings settings(shape, RVec3(position.x, position.y, position.z),
                                  Quat(rotation.w, rotation.x, rotation.y, rotation.z),
                                  EMotionType::Dynamic, Layers::DYNAMIC);
    
    settings.mMassPropertiesOverride.mMass = mass;
    settings.mOverrideMassProperties = EOverrideMassProperties::CalculateInertia;
    settings.mFriction = 0.8f;
    settings.mRestitution = 0.0f;
    settings.mLinearDamping = 1.0f;
    settings.mAngularDamping = 1.0f;
    settings.mMotionQuality = EMotionQuality::LinearCast;
    
    BodyID bodyID = gBodyInterface->CreateAndAddBody(settings, EActivation::Activate);
    gSpiderBodies.push_back({ bodyID, name, color });
    return bodyID;
}

// ============================================================================
// Joints
// ============================================================================
HingeConstraint* CreateHingeJoint(BodyID body1, BodyID body2, const glm::vec3& position,
                                   const glm::vec3& axis, float minAngle, float maxAngle,
                                   float motorFreq, float motorDamp,
                                   float maxTorque, bool motorEnabled) {
    HingeConstraintSettings settings;
    settings.mPoint1 = settings.mPoint2 = RVec3(position.x, position.y, position.z);
    settings.mHingeAxis1 = settings.mHingeAxis2 = Vec3(axis.x, axis.y, axis.z);
    settings.mNormalAxis1 = settings.mNormalAxis2 = Vec3(axis.y, -axis.x, 0).Normalized();
    
    settings.mLimitsMin = minAngle;
    settings.mLimitsMax = maxAngle;
    
    if (motorEnabled) {
        settings.mMotorSettings.mSpringSettings.mFrequency = motorFreq;
        settings.mMotorSettings.mSpringSettings.mDamping = motorDamp;
        settings.mMotorSettings.mMinTorqueLimit = -maxTorque;
        settings.mMotorSettings.mMaxTorqueLimit = maxTorque;
    }
    
    Constraint* constraint = gBodyInterface->CreateConstraint(settings, body1, body2);
    gPhysics->AddConstraint(constraint);
    return static_cast<HingeConstraint*>(constraint);
}

// ============================================================================
// Spider Build
// ============================================================================
void BuildSpider() {
    std::cout << "Building spider robot..." << std::endl;
    
    // Create central body (box)
    auto centralShape = new BoxShape(Vec3(CENTRAL_BOX_HALF_X, CENTRAL_BOX_HALF_Y, CENTRAL_BOX_HALF_Z));
    BodyID centralBody = CreateBody("body_central", glm::vec3(0, 0.6f, 0),
                                    glm::angleAxis(0.0f, glm::vec3(0,0,1)),
                                    centralShape, CENTRAL_MASS, glm::vec3(0.2f, 0.2f, 0.8f));
    
    // Create legs
    for (int leg = 0; leg < NUM_LEGS; ++leg) {
        float legX = legPositions[leg].x;
        float legY = legPositions[leg].y;
        
        // Calculate rotation to point leg outward from center
        float angle = std::atan2(legY, legX) + (legX < 0 ? JPH_PI : 0.0f);
        glm::quat coxiaRot = glm::angleAxis(angle, glm::vec3(0, 0, 1));
        glm::quat liftRot = glm::angleAxis(0.3f, glm::vec3(1, 0, 0)); // ~17 degrees
        
        // Coxia (first segment) - connects to central body
        auto coxiaShape = new CapsuleShape(COXIA_RADIUS, COXIA_HALF_HEIGHT);
        glm::vec3 coxiaPos(legX * 0.5f, legY * 0.5f, -COXIA_HALF_HEIGHT);
        BodyID coxiaID = CreateBody("leg" + std::to_string(leg) + "_coxia", coxiaPos, coxiaRot,
                                    coxiaShape, COXIA_MASS, glm::vec3(0.8f, 0.2f, 0.2f));
        
        // Femur (second segment) - extends forward/down
        glm::vec3 femurPos = coxiaPos + glm::vec3(std::cos(angle) * 0.2f, std::sin(angle) * 0.2f, -FEMUR_HALF_HEIGHT);
        auto femurShape = new CapsuleShape(FEMUR_RADIUS, FEMUR_HALF_HEIGHT);
        BodyID femurID = CreateBody("leg" + std::to_string(leg) + "_femur", femurPos, liftRot,
                                    femurShape, FEMUR_MASS, glm::vec3(0.8f, 0.6f, 0.2f));
        
        // Tibia (third segment) - extends further
        glm::vec3 tibiaPos = femurPos + glm::vec3(std::cos(angle) * 0.15f, std::sin(angle) * 0.15f, -TIBIA_HALF_HEIGHT * 2.0f);
        auto tibiaShape = new CapsuleShape(TIBIA_RADIUS, TIBIA_HALF_HEIGHT);
        BodyID tibiaID = CreateBody("leg" + std::to_string(leg) + "_tibia", tibiaPos, liftRot,
                                    tibiaShape, TIBIA_MASS, glm::vec3(0.2f, 0.8f, 0.2f));
        
        // Create joints with motors
        // 1. Hip joint (central body -> coxia) - swing left/right (Z-axis)
        HingeConstraint* joint1 = CreateHingeJoint(centralBody, coxiaID, coxiaPos,
                                                   glm::vec3(0, 0, 1), -0.6f, 0.6f, 12.0f, 1.8f, 300.0f, true);
        gSpiderJoints.push_back({ joint1, "leg" + std::to_string(leg) + "_hip", 0.0f, -0.6f, 0.6f });
        
        // 2. Knee joint (coxia -> femur) - lift up/down (Y-axis)
        glm::vec3 kneePos = coxiaPos + glm::vec3(std::cos(angle) * 0.15f, std::sin(angle) * 0.15f, -COXIA_HALF_HEIGHT);
        HingeConstraint* joint2 = CreateHingeJoint(coxiaID, femurID, kneePos,
                                                   glm::vec3(0, 1, 0), -0.5f, 1.2f, 15.0f, 2.0f, 500.0f, true);
        gSpiderJoints.push_back({ joint2, "leg" + std::to_string(leg) + "_knee", 0.0f, -0.5f, 1.2f });
        
        // 3. Ankle joint (femur -> tibia) - lower leg extension (Y-axis)
        HingeConstraint* joint3 = CreateHingeJoint(femurID, tibiaID, femurPos,
                                                   glm::vec3(0, 1, 0), -0.8f, 0.3f, 15.0f, 2.0f, 400.0f, true);
        gSpiderJoints.push_back({ joint3, "leg" + std::to_string(leg) + "_ankle", 0.0f, -0.8f, 0.3f });
    }
    
    std::cout << "Spider built with " << gSpiderBodies.size() << " bodies and " << gSpiderJoints.size() << " joints" << std::endl;
    if (gSpiderBodies.size() != 1 + NUM_LEGS * 3) {
        std::cout << "WARNING: Expected " << (1 + NUM_LEGS * 3) << " bodies, got " << gSpiderBodies.size() << std::endl;
    }
}

// ============================================================================
// Rendering
// ============================================================================
void DrawCapsule(const glm::vec3& pos, const glm::quat& rot, float radius, float halfHeight, const glm::vec3& color) {
    glm::mat4 transform = glm::translate(glm::mat4(1.0f), pos) * glm::mat4_cast(rot);
    
    glPushMatrix();
    glMultMatrixf(glm::value_ptr(transform));
    
    glColor3f(color.x, color.y, color.z);
    
    // Draw capsule as cylinder with sphere caps
    GLUquadric* quad = gluNewQuadric();
    
    // Cylinder
    glPushMatrix();
    glTranslatef(0, 0, -halfHeight);
    gluCylinder(quad, radius, radius, halfHeight * 2.0f, 16, 1);
    glPopMatrix();
    
    // Top sphere
    glPushMatrix();
    glTranslatef(0, 0, halfHeight);
    gluSphere(quad, radius, 16, 16);
    glPopMatrix();
    
    // Bottom sphere
    glPushMatrix();
    glTranslatef(0, 0, -halfHeight);
    gluSphere(quad, radius, 16, 16);
    glPopMatrix();
    
    gluDeleteQuadric(quad);
    glPopMatrix();
}

void DrawBox(const glm::vec3& pos, const glm::quat& rot, const glm::vec3& halfExtents, const glm::vec3& color) {
    glm::mat4 transform = glm::translate(glm::mat4(1.0f), pos) * glm::mat4_cast(rot);
    
    glPushMatrix();
    glMultMatrixf(glm::value_ptr(transform));
    
    glColor3f(color.x, color.y, color.z);
    
    const float x = halfExtents.x;
    const float y = halfExtents.y;
    const float z = halfExtents.z;
    
    glBegin(GL_QUADS);
    // Front
    glNormal3f(0, 0, 1);
    glVertex3f(-x, -y, z); glVertex3f(x, -y, z); glVertex3f(x, y, z); glVertex3f(-x, y, z);
    // Back
    glNormal3f(0, 0, -1);
    glVertex3f(x, -y, -z); glVertex3f(-x, -y, -z); glVertex3f(-x, y, -z); glVertex3f(x, y, -z);
    // Left
    glNormal3f(-1, 0, 0);
    glVertex3f(-x, -y, -z); glVertex3f(-x, -y, z); glVertex3f(-x, y, z); glVertex3f(-x, y, -z);
    // Right
    glNormal3f(1, 0, 0);
    glVertex3f(x, -y, z); glVertex3f(x, -y, -z); glVertex3f(x, y, -z); glVertex3f(x, y, z);
    // Top
    glNormal3f(0, 1, 0);
    glVertex3f(-x, y, z); glVertex3f(x, y, z); glVertex3f(x, y, -z); glVertex3f(-x, y, -z);
    // Bottom
    glNormal3f(0, -1, 0);
    glVertex3f(-x, -y, -z); glVertex3f(x, -y, -z); glVertex3f(x, -y, z); glVertex3f(-x, -y, z);
    glEnd();
    
    glPopMatrix();
}

void RenderPhysics() {
    // Render all spider bodies
    for (const auto& body : gSpiderBodies) {
        if (body.body.IsInvalid()) continue;
        
        Vec3 pos = gBodyInterface->GetPosition(body.body);
        Quat rot = gBodyInterface->GetRotation(body.body);
        
        glm::vec3 glPos(pos.GetX(), pos.GetY(), pos.GetZ());
        // Jolt Quat stores (w, x, y, z), same as glm
        glm::quat glRot(rot.w, rot.x, rot.y, rot.z);
        
        const Shape* shape = gBodyInterface->GetShape(body.body);
        if (!shape) continue;
        
        auto subType = shape->GetSubType();
        
        if (subType == EShapeSubType::Box) {
            const auto* box = static_cast<const BoxShape*>(shape);
            Vec3 half = box->GetHalfExtent();
            DrawBox(glPos, glRot, glm::vec3(half.GetX(), half.GetY(), half.GetZ()), body.color);
        } else if (subType == EShapeSubType::Capsule) {
            const auto* capsule = static_cast<const CapsuleShape*>(shape);
            float radius = capsule->GetRadius();
            float halfHeight = capsule->GetHalfHeight();
            DrawCapsule(glPos, glRot, radius, halfHeight, body.color);
        } else if (subType == EShapeSubType::Sphere) {
            const auto* sphere = static_cast<const SphereShape*>(shape);
            float radius = sphere->GetRadius();
            DrawCapsule(glPos, glRot, radius, 0.0f, body.color);
        }
    }
    
    // Draw ground
    glColor3f(0.3f, 0.3f, 0.35f);
    glBegin(GL_QUADS);
    glNormal3f(0, 1, 0);
    glVertex3f(-20, -0.5, -20);
    glVertex3f(20, -0.5, -20);
    glVertex3f(20, -0.5, 20);
    glVertex3f(-20, -0.5, 20);
    glEnd();
}

void DrawUI() {
    ImGui::Begin("Spider Control");
    ImGui::Text("Time: %.2fs", gTime);
    ImGui::Text("Bodies: %zu", gSpiderBodies.size());
    ImGui::Text("Joints: %zu", gSpiderJoints.size());
    ImGui::Separator();
    
    ImGui::Text("Leg Motor Targets");
    for (size_t i = 0; i < gSpiderJoints.size(); i++) {
        auto& joint = gSpiderJoints[i];
        ImGui::SliderFloat(joint.name.c_str(), &joint.targetAngle, joint.minAngle, joint.maxAngle);
    }
    
    ImGui::Separator();
    ImGui::Text("Camera Controls:");
    ImGui::Text("Right Mouse + Drag: Rotate");
    ImGui::Text("Scroll: Zoom");
    
    ImGui::End();
}

void UpdateJoints() {
    float timestep = 1.0f / 60.0f;
    for (auto& joint : gSpiderJoints) {
        if (joint.constraint && !joint.constraint->IsDisabled()) {
            joint.constraint->SetMotorTargetAndResume(joint.targetAngle, timestep);
        }
    }
}

void ProcessKeyboard(GLFWwindow* window) {
    if (glfwGetKey(window, GLFW_KEY_ESCAPE) == GLFW_PRESS) {
        glfwSetWindowShouldClose(window, true);
    }
    
    // Reset all joints to neutral
    if (glfwGetKey(window, GLFW_KEY_R) == GLFW_PRESS) {
        for (auto& joint : gSpiderJoints) {
            joint.targetAngle = 0.0f;
        }
    }
    
    // Wave animation
    static float wavePhase = 0.0f;
    if (glfwGetKey(window, GLFW_KEY_W) == GLFW_PRESS) {
        wavePhase += 0.05f;
        for (int leg = 0; leg < NUM_LEGS; leg++) {
            float offset = leg * (JPH_PI * 2.0f / NUM_LEGS);
            // Hip: wave
            gSpiderJoints[leg * 3 + 0].targetAngle = 0.3f * std::sin(wavePhase + offset);
            // Knee: lift
            gSpiderJoints[leg * 3 + 1].targetAngle = 0.2f + 0.3f * std::abs(std::sin(wavePhase + offset));
            // Ankle: extend
            gSpiderJoints[leg * 3 + 2].targetAngle = -0.2f * std::sin(wavePhase + offset);
        }
    }
    
    // Stand up gesture
    if (glfwGetKey(window, GLFW_KEY_U) == GLFW_PRESS) {
        for (int leg = 0; leg < NUM_LEGS; leg++) {
            gSpiderJoints[leg * 3 + 1].targetAngle = 0.8f; // Knee lift
            gSpiderJoints[leg * 3 + 2].targetAngle = 0.0f; // Ankle
        }
    }
}

int main() {
    std::cout << "=== Spider Robot Viewer ===" << std::endl;
    
    // Initialize GLFW
    if (!glfwInit()) {
        std::cerr << "GLFW init failed!" << std::endl;
        return 1;
    }
    
    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3);
    glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);
    
    GLFWwindow* window = glfwCreateWindow(1280, 720, "Spider Robot Viewer", nullptr, nullptr);
    if (!window) {
        std::cerr << "Window creation failed!" << std::endl;
        glfwTerminate();
        return 1;
    }
    
    glfwMakeContextCurrent(window);
    glfwSwapInterval(1); // VSync
    
    // Initialize GLEW
    glewExperimental = GL_TRUE;
    if (glewInit() != GLEW_OK) {
        std::cerr << "GLEW init failed!" << std::endl;
        return 1;
    }
    
    glEnable(GL_DEPTH_TEST);
    glEnable(GL_LIGHTING);
    glEnable(GL_LIGHT0);
    glEnable(GL_COLOR_MATERIAL);
    glColorMaterial(GL_FRONT_AND_BACK, GL_AMBIENT_AND_DIFFUSE);
    
    // Initialize ImGui
    IMGUI_CHECKVERSION();
    ImGui::CreateContext();
    ImGuiIO& io = ImGui::GetIO();
    io.DisplaySize = ImVec2(1280, 720);
    
    ImGui_ImplGlfw_InitForOpenGL(window, true);
    ImGui_ImplOpenGL3_Init("#version 330");
    
    // Initialize physics
    InitJolt();
    CreateArena();
    BuildSpider();
    
    std::cout << "Created " << gSpiderBodies.size() << " bodies" << std::endl;
    std::cout << "Created " << gSpiderJoints.size() << " motorized joints" << std::endl;
    
    // Mouse callbacks
    glfwSetCursorPosCallback(window, [](GLFWwindow* w, double x, double y) {
        if (glfwGetMouseButton(w, GLFW_MOUSE_BUTTON_RIGHT) == GLFW_PRESS) {
            if (glfwGetKey(w, GLFW_KEY_LEFT_CONTROL) == GLFW_PRESS || glfwGetKey(w, GLFW_KEY_RIGHT_CONTROL) == GLFW_PRESS) {
                // Camera rotation
                float dx = (float)(x - gLastMouseX) * 0.01f;
                float dy = (float)(y - gLastMouseY) * 0.01f;
                gCameraYaw += dx;
                gCameraPitch += dy;
                gCameraPitch = glm::clamp(gCameraPitch, -1.5f, 1.5f);
            }
        }
        gLastMouseX = (float)x;
        gLastMouseY = (float)y;
    });
    
    glfwSetScrollCallback(window, [](GLFWwindow* w, double xoffset, double yoffset) {
        gCameraDistance *= (float)std::pow(1.1, -yoffset * 0.5f);
        gCameraDistance = glm::clamp(gCameraDistance, 3.0f, 50.0f);
    });
    
    glfwSetMouseButtonCallback(window, [](GLFWwindow* w, int button, int action, int mods) {
        if (button == GLFW_MOUSE_BUTTON_RIGHT) {
            if (action == GLFW_PRESS) {
                glfwGetCursorPos(w, &gLastMouseX, &gLastMouseY);
            }
        }
    });
    
    // Main loop
    double lastTime = glfwGetTime();
    int frameCount = 0;
    
    while (!glfwWindowShouldClose(window)) {
        double currentTime = glfwGetTime();
        float deltaTime = (float)(currentTime - lastTime);
        lastTime = currentTime;
        gTime += deltaTime;
        
        // Process input
        glfwPollEvents();
        ProcessKeyboard(window);
        
        // Apply motor targets
        UpdateJoints();
        
        // Step physics with fixed timestep
        float fixedDeltaTime = 1.0f / 60.0f;
        int steps = (int)(deltaTime / fixedDeltaTime) + 1;
        gPhysics->Update(fixedDeltaTime, steps, *gTempAllocator, *gJobSystem);
        
        // Render
        glClearColor(0.1f, 0.1f, 0.15f, 1.0f);
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
        
        glMatrixMode(GL_PROJECTION);
        glLoadIdentity();
        gluPerspective(60.0f, 1280.0f / 720.0f, 0.1f, 1000.0f);
        
        glMatrixMode(GL_MODELVIEW);
        glLoadIdentity();
        
        float camX = gCameraDistance * cosf(gCameraPitch) * sinf(gCameraYaw);
        float camY = gCameraDistance * sinf(gCameraPitch) + 1.5f;
        float camZ = gCameraDistance * cosf(gCameraPitch) * cosf(gCameraYaw);
        gluLookAt(camX, camY, camZ, 0.0f, 0.5f, 0.0f, 0.0f, 1.0f, 0.0f);
        
        // Lighting
        GLfloat lightPos[] = { 5.0f, 10.0f, 5.0f, 1.0f };
        GLfloat lightAmb[] = { 0.3f, 0.3f, 0.3f, 1.0f };
        GLfloat lightDiff[] = { 0.8f, 0.8f, 0.8f, 1.0f };
        glLightfv(GL_LIGHT0, GL_POSITION, lightPos);
        glLightfv(GL_LIGHT0, GL_AMBIENT, lightAmb);
        glLightfv(GL_LIGHT0, GL_DIFFUSE, lightDiff);
        
        RenderPhysics();
        
        // ImGui
        ImGui_ImplOpenGL3_NewFrame();
        ImGui_ImplGlfw_NewFrame();
        ImGui::NewFrame();
        
        DrawUI();
        
        ImGui::Render();
        ImGui_ImplOpenGL3_RenderDrawData(ImGui::GetDrawData());
        
        glfwSwapBuffers(window);
        
        frameCount++;
        if (frameCount % 60 == 0) {
            std::cout << "FPS: " << (1.0 / deltaTime) << std::endl;
        }
    }
    
    // Cleanup
    std::cout << "Shutting down..." << std::endl;
    
    ImGui_ImplOpenGL3_Shutdown();
    ImGui_ImplGlfw_Shutdown();
    ImGui::DestroyContext();
    
    // Cleanup physics
    // Remove constraints first
    for (auto& joint : gSpiderJoints) {
        if (joint.constraint) {
            gPhysics->RemoveConstraint(joint.constraint);
            delete joint.constraint;
        }
    }
    
    // Remove all bodies
    for (auto& body : gSpiderBodies) {
        gBodyInterface->RemoveBody(body.body);
    }
    
    // Cleanup Jolt
    delete gPhysics;
    delete gTempAllocator;
    delete gJobSystem;
    if (JPH::Factory::sInstance) {
        delete JPH::Factory::sInstance;
        JPH::Factory::sInstance = nullptr;
    }
    
    glfwDestroyWindow(window);
    glfwTerminate();
    
    return 0;
}
