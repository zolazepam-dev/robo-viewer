#include <Jolt/Jolt.h>
#include "RobotLoader.h"

#include <cmath>
#include <fstream>
#include <iostream>
#include <map>
#include <unordered_map>
#include <sstream>
#include <algorithm>
#include <filesystem>

#include <nlohmann/json.hpp>
#include <tinyxml2.h>

#include <Jolt/Math/Math.h>
#include <Jolt/Physics/Body/BodyCreationSettings.h>
#include <Jolt/Physics/Body/BodyInterface.h>
#include <Jolt/Physics/Collision/Shape/BoxShape.h>
#include <Jolt/Physics/Collision/Shape/SphereShape.h>
#include <Jolt/Physics/Collision/Shape/CylinderShape.h>
#include <Jolt/Physics/Collision/Shape/CapsuleShape.h>
#include <Jolt/Physics/Collision/Shape/ConvexHullShape.h>
#include <Jolt/Physics/Collision/Shape/MeshShape.h>
#include <Jolt/Physics/Collision/Shape/StaticCompoundShape.h>
#include <Jolt/Physics/Collision/Shape/RotatedTranslatedShape.h>
#include <Jolt/Physics/Collision/Shape/ScaledShape.h>
#include <Jolt/Physics/Collision/Shape/OffsetCenterOfMassShape.h>
#include <Jolt/Physics/Constraints/HingeConstraint.h>
#include <Jolt/Physics/Constraints/SliderConstraint.h>

#include "PhysicsCore.h"

namespace {

// Lightweight default resolver for geom/joint attributes (single inheritance)
struct MJCFDefaults
{
    // class name -> map of attr key->value (string raw)
    std::unordered_map<std::string, std::map<std::string, std::string>> geomDefaults;
    std::unordered_map<std::string, std::map<std::string, std::string>> jointDefaults;

    void Ingest(const tinyxml2::XMLElement* defaults)
    {
        for (const tinyxml2::XMLElement* d = defaults; d; d = d->NextSiblingElement("default")) {
            const char* cls = d->Attribute("class");
            std::string name = cls ? cls : "";
            // geom child
            if (const tinyxml2::XMLElement* g = d->FirstChildElement("geom")) {
                auto& bucket = geomDefaults[name];
                for (const tinyxml2::XMLAttribute* a = g->FirstAttribute(); a; a = a->Next()) {
                    bucket[a->Name()] = a->Value();
                }
            }
            // joint child
            if (const tinyxml2::XMLElement* j = d->FirstChildElement("joint")) {
                auto& bucket = jointDefaults[name];
                for (const tinyxml2::XMLAttribute* a = j->FirstAttribute(); a; a = a->Next()) {
                    bucket[a->Name()] = a->Value();
                }
            }
        }
    }

    // Merge class defaults onto element attributes; element wins
    void ApplyGeomDefaults(const tinyxml2::XMLElement* elem, std::map<std::string, std::string>& out) const
    {
        const char* cls = elem->Attribute("class");
        if (cls) {
            auto it = geomDefaults.find(cls);
            if (it != geomDefaults.end()) {
                out.insert(it->second.begin(), it->second.end());
            }
        }
        for (const tinyxml2::XMLAttribute* a = elem->FirstAttribute(); a; a = a->Next()) {
            out[a->Name()] = a->Value();
        }
    }

    void ApplyJointDefaults(const tinyxml2::XMLElement* elem, std::map<std::string, std::string>& out) const
    {
        const char* cls = elem->Attribute("class");
        if (cls) {
            auto it = jointDefaults.find(cls);
            if (it != jointDefaults.end()) {
                out.insert(it->second.begin(), it->second.end());
            }
        }
        for (const tinyxml2::XMLAttribute* a = elem->FirstAttribute(); a; a = a->Next()) {
            out[a->Name()] = a->Value();
        }
    }
};

struct MJCFBodyInstance
{
    std::string name;
    JPH::Ref<JPH::Shape> shape;
    float mass = 1.0f;
    JPH::RVec3 worldPos {0, 0, 0};
    JPH::Quat worldRot = JPH::Quat::sIdentity();
    bool isStatic = false;
};

struct MeshData
{
    std::vector<JPH::Vec3> vertices;
    std::vector<uint32_t> indices;
};

JPH::Vec3 ParseVec3(const char* str, const JPH::Vec3& def = JPH::Vec3::sZero())
{
    if (!str) return def;
    float x=0,y=0,z=0;
    std::stringstream ss(str);
    ss >> x >> y >> z;
    if (ss.fail()) return def;
    // Convert MuJoCo (Z-up) to Jolt (Y-up): (x,y,z) -> (x,z,y)
    return JPH::Vec3(x, z, y);
}

JPH::Vec3 ParseVec3Raw(const char* str, const JPH::Vec3& def = JPH::Vec3::sZero())
{
    if (!str) return def;
    float x=0,y=0,z=0;
    std::stringstream ss(str);
    ss >> x >> y >> z;
    if (ss.fail()) return def;
    return JPH::Vec3(x, y, z);
}

JPH::Quat ParseEuler(const char* str)
{
    if (!str) return JPH::Quat::sIdentity();
    float rx=0, ry=0, rz=0;
    std::stringstream ss(str);
    ss >> rx >> ry >> rz;
    if (ss.fail()) return JPH::Quat::sIdentity();
    // Convert MuJoCo Euler (Z-up) to Jolt (Y-up): rotate around (x, z, y)
    return JPH::Quat::sEulerAngles(JPH::Vec3(JPH::DegreesToRadians(rx), JPH::DegreesToRadians(rz), JPH::DegreesToRadians(ry)));
}

// Compute rotation that aligns local Z axis with target direction
JPH::Quat RotationFromZAxis(const JPH::Vec3& targetDir)
{
    JPH::Vec3 zAxis(0, 0, 1);
    JPH::Vec3 dir = targetDir.Normalized();
    if (dir.IsNearZero()) return JPH::Quat::sIdentity();
    // Handle parallel/anti-parallel cases
    float dot = zAxis.Dot(dir);
    if (dot > 0.9999f) return JPH::Quat::sIdentity();
    if (dot < -0.9999f) return JPH::Quat::sRotation(JPH::Vec3::sAxisX(), JPH::JPH_PI);
    JPH::Vec3 axis = zAxis.Cross(dir).Normalized();
    float angle = acosf(dot);
    return JPH::Quat::sRotation(axis, angle);
}

static bool LoadBinarySTL(const std::filesystem::path& path, std::vector<JPH::Vec3>& outVerts, std::vector<uint32_t>& outIdx)
{
    std::ifstream f(path, std::ios::binary);
    if (!f.is_open()) return false;
    char header[80];
    f.read(header, 80);
    uint32_t triCount = 0;
    f.read(reinterpret_cast<char*>(&triCount), sizeof(uint32_t));
    outVerts.reserve(triCount * 3);
    outIdx.reserve(triCount * 3);
    for (uint32_t i = 0; i < triCount; ++i) {
        float n[3]; f.read(reinterpret_cast<char*>(n), sizeof(float) * 3);
        for (int v = 0; v < 3; ++v) {
            float p[3]; f.read(reinterpret_cast<char*>(p), sizeof(float) * 3);
            outVerts.emplace_back(p[0], p[1], p[2]);
            outIdx.push_back(static_cast<uint32_t>(outVerts.size() - 1));
        }
        uint16_t attr; f.read(reinterpret_cast<char*>(&attr), sizeof(uint16_t));
        if (!f.good()) return false;
    }
    return true;
}

JPH::Ref<JPH::Shape> ParseGeomShape(const tinyxml2::XMLElement* geom,
                                    const std::unordered_map<std::string, MeshData>& meshCache,
                                    bool isDynamic)
{
    const char* type = geom->Attribute("type");
    std::string t = type ? type : "box";
    const char* geomName = geom->Attribute("name");
    std::cout << "RobotLoader: ParseGeomShape name='" << (geomName ? geomName : "") << "' type=" << t << std::endl;
    if (t == "sphere") {
        float radius = geom->FloatAttribute("size", 0.5f);
        JPH::SphereShapeSettings settings(radius);
        auto res = settings.Create();
        return res.HasError() ? nullptr : res.Get();
    }
    if (t == "box") {
        JPH::Vec3 half = JPH::Vec3::sReplicate(geom->FloatAttribute("size", 0.5f));
        const char* sizeStr = geom->Attribute("size");
        if (sizeStr) {
            std::stringstream ss(sizeStr);
            float x, y, z;
            ss >> x >> y >> z;
            if (!ss.fail()) {
                half.SetX(x);
                half.SetY(y);
                half.SetZ(z);
            }
        }
        JPH::BoxShapeSettings settings(half);
        auto res = settings.Create();
        return res.HasError() ? nullptr : res.Get();
    }
    if (t == "cylinder") {
        const char* fromtoStr = geom->Attribute("fromto");
        if (fromtoStr) {
            std::cout << "RobotLoader: parsing cylinder fromto='" << fromtoStr << "'" << std::endl;
            // cylinder defined by two endpoints
            float x1, y1, z1, x2, y2, z2;
            std::stringstream ss(fromtoStr);
            ss >> x1 >> y1 >> z1 >> x2 >> y2 >> z2;
            JPH::Vec3 p1(x1, z1, y1), p2(x2, z2, y2); // MuJoCo (Z-up) to Jolt (Y-up)
            if (ss.fail()) {
                std::cerr << "RobotLoader: invalid fromto attribute: " << fromtoStr << std::endl;
                return nullptr;
            }
            JPH::Vec3 delta = p2 - p1;
            float length = delta.Length();
            if (length < 1e-6f) {
                std::cerr << "RobotLoader: fromto endpoints coincide" << std::endl;
                return nullptr;
            }
            JPH::Vec3 dir = delta / length;
            JPH::Vec3 midpoint = (p1 + p2) * 0.5f;
            float radius = geom->FloatAttribute("size", 0.1f); // size is radius
            float halfHeight = length * 0.5f;
            JPH::CylinderShapeSettings cylSettings(halfHeight, radius);
            auto cylRes = cylSettings.Create();
            if (cylRes.HasError()) return nullptr;
            JPH::Ref<JPH::Shape> cylinder = cylRes.Get();
            // Rotate cylinder from Z axis to dir, then translate to midpoint
            JPH::RotatedTranslatedShapeSettings rtSettings(midpoint, RotationFromZAxis(dir), cylinder);
            auto rtRes = rtSettings.Create();
            return rtRes.HasError() ? nullptr : rtRes.Get();
        } else {
            // MuJoCo cylinder: size[0] = radius, size[1] = half-height (z)
            JPH::Vec3 size = ParseVec3(geom->Attribute("size"), JPH::Vec3(0.1f, 0.1f, 0.1f));
            float radius = size.GetX();
            float halfHeight = size.GetY();
            JPH::CylinderShapeSettings settings(halfHeight, radius);
            auto res = settings.Create();
            return res.HasError() ? nullptr : res.Get();
        }
    }
    if (t == "capsule") {
        const char* fromtoStr = geom->Attribute("fromto");
        if (fromtoStr) {
            // Capsule defined by two endpoints (line segment) and radius
            float x1, y1, z1, x2, y2, z2;
            std::stringstream ss(fromtoStr);
            ss >> x1 >> y1 >> z1 >> x2 >> y2 >> z2;
            JPH::Vec3 p1(x1, z1, y1), p2(x2, z2, y2); // MuJoCo (Z-up) to Jolt (Y-up)
            if (ss.fail()) {
                std::cerr << "RobotLoader: invalid capsule fromto: " << fromtoStr << std::endl;
                return nullptr;
            }
            JPH::Vec3 delta = p2 - p1;
            float length = delta.Length();
            if (length < 1e-6f) {
                // Degenerate capsule - use sphere
                float radius = geom->FloatAttribute("size", 0.1f);
                JPH::SphereShapeSettings settings(radius);
                auto res = settings.Create();
                return res.HasError() ? nullptr : res.Get();
            }
            JPH::Vec3 dir = delta.Normalized();
            JPH::Vec3 midpoint = (p1 + p2) * 0.5f;
            float radius = geom->FloatAttribute("size", 0.1f);
            float halfHeight = length * 0.5f;
            JPH::CapsuleShapeSettings capsuleSettings(halfHeight, radius);
            auto capsuleRes = capsuleSettings.Create();
            if (capsuleRes.HasError()) return nullptr;
            JPH::Ref<JPH::Shape> capsule = capsuleRes.Get();
            // Rotate capsule from Y axis to dir, then translate to midpoint
            JPH::RotatedTranslatedShapeSettings rtSettings(midpoint, RotationFromZAxis(dir), capsule);
            auto rtRes = rtSettings.Create();
            return rtRes.HasError() ? nullptr : rtRes.Get();
        } else {
            // Simple capsule along local Y axis
            JPH::Vec3 size = ParseVec3(geom->Attribute("size"), JPH::Vec3(0.1f, 0.1f, 0.1f));
            float radius = size.GetX();
            float halfHeight = size.GetY();
            JPH::CapsuleShapeSettings settings(halfHeight, radius);
            auto res = settings.Create();
            return res.HasError() ? nullptr : res.Get();
        }
    }
    if (t == "plane") {
        // MuJoCo plane: size="x y z" where x,y are half-widths, z is ignored (zero thickness)
        // Approximate with a thin box of thickness 0.01
        JPH::Vec3 size = ParseVec3(geom->Attribute("size"), JPH::Vec3(1.0f, 1.0f, 0.0f));
        float halfWidth = size.GetX();
        float halfDepth = size.GetY();
        float halfThickness = 0.005f; // very thin
        JPH::BoxShapeSettings settings(JPH::Vec3(halfWidth, halfDepth, halfThickness));
        auto res = settings.Create();
        return res.HasError() ? nullptr : res.Get();
    }
    if (t == "mesh") {
        const char* meshName = geom->Attribute("mesh");
        if (!meshName) {
            std::cerr << "RobotLoader: mesh geom missing mesh attribute" << std::endl;
            return nullptr;
        }
        auto it = meshCache.find(meshName);
        if (it == meshCache.end() || it->second.vertices.empty()) {
            std::cerr << "RobotLoader: mesh asset not found or empty: " << meshName << std::endl;
            return nullptr;
        }
        const MeshData& md = it->second;
        if (isDynamic) {
            JPH::ConvexHullShapeSettings hull(md.vertices.data(), md.vertices.size());
            auto res = hull.Create();
            if (!res.HasError()) return res.Get();
            std::cerr << "RobotLoader: hull failed for mesh " << meshName << " falling back sphere" << std::endl;
            JPH::SphereShapeSettings settings(0.05f);
            auto res2 = settings.Create();
            return res2.HasError() ? nullptr : res2.Get();
        } else {
            // Convert to Jolt's format
            JPH::VertexList vertices;
            for (const JPH::Vec3& v : md.vertices) {
                vertices.push_back(JPH::Float3(v.GetX(), v.GetY(), v.GetZ()));
            }

            JPH::IndexedTriangleList triangles;
            for (size_t i = 0; i < md.indices.size(); i += 3) {
                if (i + 2 < md.indices.size()) {
                    triangles.push_back(JPH::IndexedTriangle(
                        static_cast<uint32_t>(md.indices[i]),
                        static_cast<uint32_t>(md.indices[i + 1]),
                        static_cast<uint32_t>(md.indices[i + 2])
                    ));
                }
            }

            JPH::MeshShapeSettings tri(vertices, triangles);
            auto res = tri.Create();
            if (!res.HasError()) return res.Get();
            std::cerr << "RobotLoader: tri mesh failed for mesh " << meshName << " falling back sphere" << std::endl;
            JPH::SphereShapeSettings settings(0.05f);
            auto res2 = settings.Create();
            return res2.HasError() ? nullptr : res2.Get();
        }
    }
    std::cerr << "RobotLoader: Unsupported MJCF geom type: " << t << std::endl;
    return nullptr;
}

void BuildMJCFRecursive(const tinyxml2::XMLElement* bodyElem,
                        const JPH::RVec3& parentPos,
                        const JPH::Quat& parentRot,
                        JPH::BodyInterface& bodyInterface,
                        JPH::PhysicsSystem& physicsSystem,
                        const JPH::ObjectLayer dynamicLayer,
                        const std::unordered_map<std::string, MeshData>& meshCache,
                        const MJCFDefaults& defaults,
                        std::map<std::string, JPH::BodyID>& bodyMap,
                        RobotData& outData)
{
    JPH::Vec3 localPos = ParseVec3(bodyElem->Attribute("pos"));
    JPH::Quat localRot = ParseEuler(bodyElem->Attribute("euler"));

    const char* bodyName = bodyElem->Attribute("name");
    std::string name = bodyName ? bodyName : "";

    // Special handling for satellite bodies - use the position of their core geom
    // The core geom position is in the body's LOCAL frame, must be rotated by body's euler
    if (name.find("_s") != std::string::npos) {
        for (const tinyxml2::XMLElement* geom = bodyElem->FirstChildElement("geom"); geom; geom = geom->NextSiblingElement("geom")) {
            const char* geomName = geom->Attribute("name");
            if (geomName && std::string(geomName).find("core") != std::string::npos) {
                JPH::Vec3 corePosLocal = ParseVec3(geom->Attribute("pos"));
                if (!corePosLocal.IsNearZero()) {
                    // Rotate core position by body's local rotation to get effective local position
                    localPos = localRot * corePosLocal;
                    break;
                }
            }
        }
    }
    
    JPH::RVec3 worldPos = parentPos + parentRot * JPH::RVec3(localPos);
    JPH::Quat worldRot = parentRot * localRot;

    struct SubShape {
        JPH::Ref<JPH::Shape> shape;
        JPH::Vec3 pos;
        JPH::Quat rot;
    };
    std::vector<SubShape> subShapes;
    
    // MJCF mass handling: body mass overrides geom masses; if neither, default to 1.0
    float bodyMass = -1.0f;
    if (bodyElem->Attribute("mass")) {
        bodyMass = bodyElem->FloatAttribute("mass");
    }
    
    float totalGeomMass = 0.0f;
    bool hasGeomMass = false;
    
    // Process all geoms
    for (const tinyxml2::XMLElement* geom = bodyElem->FirstChildElement("geom"); geom; geom = geom->NextSiblingElement("geom")) {
        // Apply defaults to geom
        std::map<std::string, std::string> geomAttrs;
        defaults.ApplyGeomDefaults(geom, geomAttrs);
        
        float geomMass = -1.0f;
        auto massIt = geomAttrs.find("mass");
        if (massIt != geomAttrs.end()) {
            try {
                geomMass = std::stof(massIt->second);
            } catch (...) {
                geomMass = -1.0f;
            }
        }
        
        if (geomMass > 0.0f) {
            totalGeomMass += geomMass;
            hasGeomMass = true;
        }
        
        // Parse geom local transform
        JPH::Vec3 geomPos = ParseVec3(geom->Attribute("pos"));
        JPH::Quat geomRot = ParseEuler(geom->Attribute("euler"));
        JPH::Ref<JPH::Shape> geomShape = ParseGeomShape(geom, meshCache, true);
        if (geomShape) {
            subShapes.push_back({geomShape, geomPos, geomRot});
        }
    }
    
    // Determine final mass: body mass > geom masses > default
    float finalMass = (bodyMass >= 0.0f) ? bodyMass : (hasGeomMass ? totalGeomMass : 1.0f);
    
    JPH::Ref<JPH::Shape> shape;
    if (subShapes.empty()) {
        // default tiny sphere
        JPH::SphereShapeSettings s(0.1f);
        auto res = s.Create();
        shape = res.HasError() ? nullptr : res.Get();
    } else if (subShapes.size() == 1) {
        // single geom: always create RotatedTranslatedShape (handles zero offset)
        const SubShape& sub = subShapes.front();
        JPH::RotatedTranslatedShapeSettings rtSettings(sub.pos, sub.rot, sub.shape);
        auto rtRes = rtSettings.Create();
        shape = rtRes.HasError() ? nullptr : rtRes.Get();
    } else {
        // multiple geoms: create StaticCompoundShape
        JPH::StaticCompoundShapeSettings comp;
        for (const auto& sub : subShapes) {
            comp.AddShape(sub.pos, sub.rot, sub.shape);
        }
        auto compRes = comp.Create();
        shape = compRes.HasError() ? nullptr : compRes.Get();
    }

    // Freejoint or no joints => root is dynamic unless mass==0
    (void)bodyElem->FirstChildElement("freejoint"); // unused
    if (!shape) {
        // default tiny sphere to keep hierarchy valid
        JPH::SphereShapeSettings s(0.1f);
        auto res = s.Create();
        shape = res.HasError() ? nullptr : res.Get();
    }
    if (shape) {
        JPH::BodyCreationSettings settings(shape,
                                           worldPos,
                                           worldRot,
                                           finalMass > 0.0f ? JPH::EMotionType::Dynamic : JPH::EMotionType::Static,
                                           finalMass > 0.0f ? dynamicLayer : Layers::STATIC);
        if (finalMass > 0.0f) {
            settings.mOverrideMassProperties = JPH::EOverrideMassProperties::CalculateInertia;
            settings.mMassPropertiesOverride.mMass = finalMass;
            
            // Disable collision between bodies of the same robot to prevent self-collision explosions
            // All satellite bodies of the same robot should not collide with each other
            settings.mCollisionGroup.SetGroupID(1);  // Robot body group
            settings.mCollisionGroup.SetSubGroupID(0);  // Same sub-group = no self-collision
        }
        JPH::Body* body = bodyInterface.CreateBody(settings);
        if (body) {
            bodyInterface.AddBody(body->GetID(), finalMass > 0.0f ? JPH::EActivation::Activate : JPH::EActivation::DontActivate);
            if (!name.empty()) bodyMap[name] = body->GetID();
            outData.bodies.push_back(body->GetID());
            std::cout << "RobotLoader: Created body '" << name << "' at (" << worldPos.GetX() << ", " << worldPos.GetY() << ", " << worldPos.GetZ() << ")" << std::endl;
        }
    }

    // Constraints for joints declared in children referencing this as parent
    const tinyxml2::XMLElement* child = bodyElem->FirstChildElement("body");
    for (; child; child = child->NextSiblingElement("body")) {
        BuildMJCFRecursive(child, worldPos, worldRot, bodyInterface, physicsSystem, dynamicLayer, meshCache, defaults, bodyMap, outData);

        // Find joints inside child (hinge/slide)
        const char* childName = child->Attribute("name");
        std::string childNameStr = childName ? childName : "";
        auto childIt = bodyMap.find(childNameStr);
        if (childIt == bodyMap.end()) continue;
        
        // Get the actual world position and rotation of the child body after it's been created
        JPH::RVec3 childWorldPos = bodyInterface.GetPosition(childIt->second);
        JPH::Quat childWorldRot = bodyInterface.GetRotation(childIt->second);
        
        // Get the parent body's world position and rotation
        auto parentIt = bodyMap.find(name);
        JPH::RVec3 parentWorldPos = parentIt != bodyMap.end() ? bodyInterface.GetPosition(parentIt->second) : worldPos;
        JPH::Quat parentWorldRot = parentIt != bodyMap.end() ? bodyInterface.GetRotation(parentIt->second) : worldRot;
        
        for (const tinyxml2::XMLElement* joint = child->FirstChildElement("joint"); joint; joint = joint->NextSiblingElement("joint")) {
            // Apply defaults to joint
            std::map<std::string, std::string> jointAttrs;
            defaults.ApplyJointDefaults(joint, jointAttrs);
            
            const char* jtype = joint->Attribute("type");
            std::string jt = jtype ? jtype : "hinge";
            
            // Get joint position (default to child body position if not specified)
            JPH::Vec3 jointPos = ParseVec3(joint->Attribute("pos"));
            JPH::RVec3 anchor = childWorldPos + childWorldRot * JPH::RVec3(jointPos);
            
            JPH::Vec3 axis = ParseVec3(joint->Attribute("axis"), JPH::Vec3(0, 0, 1));
            if (axis.IsNearZero()) axis = JPH::Vec3::sAxisZ();
            axis = (childWorldRot * axis).Normalized();

            if (parentIt == bodyMap.end()) {
                std::cout << "RobotLoader: Could not find parent body '" << name << "' for joint in child '" << childNameStr << "'" << std::endl;
                continue;
            }
            
            std::cout << "RobotLoader: Creating " << jt << " joint between parent '" << name << "' and child '" << childNameStr 
                      << "' at anchor (" << anchor.GetX() << ", " << anchor.GetY() << ", " << anchor.GetZ() << ")" << std::endl;
            
            if (jt == "hinge") {
                JPH::HingeConstraintSettings settings;
                settings.mSpace = JPH::EConstraintSpace::WorldSpace;
                settings.mPoint1 = anchor;
                settings.mPoint2 = anchor;
                settings.mHingeAxis1 = axis;
                settings.mHingeAxis2 = axis;
                settings.mNormalAxis1 = axis.GetNormalizedPerpendicular();
                settings.mNormalAxis2 = settings.mNormalAxis1;
                
                // Check for range in either original joint or defaults
                const char* rangeAttr = joint->Attribute("range");
                auto rangeIt = jointAttrs.find("range");
                std::string rangeStr = rangeAttr ? rangeAttr : (rangeIt != jointAttrs.end() ? rangeIt->second : "");
                
                if (!rangeStr.empty()) {
                    float lo=0, hi=0; 
                    std::stringstream ss(rangeStr); 
                    ss >> lo >> hi;
                    if (!ss.fail()) { 
                        settings.mLimitsMin = JPH::DegreesToRadians(lo); 
                        settings.mLimitsMax = JPH::DegreesToRadians(hi); 
                    }
                }
                
                JPH::TwoBodyConstraint* c = bodyInterface.CreateConstraint(&settings, parentIt->second, childIt->second);
                if (c) { 
                    physicsSystem.AddConstraint(c); 
                    outData.constraints.push_back(c); 
                    std::cout << "RobotLoader: Successfully created hinge constraint" << std::endl;
                } else {
                    std::cout << "RobotLoader: Failed to create hinge constraint" << std::endl;
                }
            } else if (jt == "slide") {
                JPH::SliderConstraintSettings settings;
                settings.mSpace = JPH::EConstraintSpace::WorldSpace;
                settings.mPoint1 = anchor;
                settings.mPoint2 = anchor;
                settings.mSliderAxis1 = axis;
                settings.mSliderAxis2 = axis;
                
                // Check for range in either original joint or defaults
                const char* rangeAttr = joint->Attribute("range");
                auto rangeIt = jointAttrs.find("range");
                std::string rangeStr = rangeAttr ? rangeAttr : (rangeIt != jointAttrs.end() ? rangeIt->second : "");
                
                if (!rangeStr.empty()) {
                    float lo=0, hi=0; 
                    std::stringstream ss(rangeStr); 
                    ss >> lo >> hi;
                    if (!ss.fail()) { 
                        settings.mLimitsMin = lo; 
                        settings.mLimitsMax = hi; 
                    }
                }
                
                JPH::TwoBodyConstraint* c = bodyInterface.CreateConstraint(&settings, parentIt->second, childIt->second);
                if (c) { 
                    physicsSystem.AddConstraint(c); 
                    outData.constraints.push_back(c); 
                    std::cout << "RobotLoader: Successfully created slide constraint" << std::endl;
                } else {
                    std::cout << "RobotLoader: Failed to create slide constraint" << std::endl;
                }
            }
        }
}
}

struct MJCFContext
{
    MJCFDefaults defaults;
    std::filesystem::path baseDir;
    std::filesystem::path meshDir;
};

static const tinyxml2::XMLElement* ResolveInclude(const tinyxml2::XMLElement* mjRoot, const MJCFContext& ctx, const std::string& file, tinyxml2::XMLDocument& outDoc)
{
    std::filesystem::path inc = ctx.baseDir / file;
    if (outDoc.LoadFile(inc.string().c_str()) != tinyxml2::XML_SUCCESS) {
        std::cerr << "RobotLoader: Failed to load include MJCF " << inc << std::endl;
        return nullptr;
    }
    return outDoc.FirstChildElement("mujoco");
}

// Wrapper to gather <worldbody> bodies from root + includes
static void ProcessWorlds(const tinyxml2::XMLElement* mj,
                          JPH::BodyInterface& body_interface,
                          JPH::PhysicsSystem& physics,
                          const MJCFContext& ctx,
                          const JPH::ObjectLayer dynamicLayer,
                          const std::unordered_map<std::string, MeshData>& meshCache,
                          std::map<std::string, JPH::BodyID>& body_map,
                          RobotData& robot_data)
{
    auto process_world = [&](const tinyxml2::XMLElement* wb) {
        for (const tinyxml2::XMLElement* body = wb->FirstChildElement("body"); body; body = body->NextSiblingElement("body")) {
            BuildMJCFRecursive(body, JPH::RVec3::sZero(), JPH::Quat::sIdentity(), body_interface, physics, dynamicLayer, meshCache, ctx.defaults, body_map, robot_data);
        }
    };

    if (const tinyxml2::XMLElement* wb = mj->FirstChildElement("worldbody")) {
        process_world(wb);
    }

    // Single-level include resolution
    for (const tinyxml2::XMLElement* inc = mj->FirstChildElement("include"); inc; inc = inc->NextSiblingElement("include")) {
        const char* file = inc->Attribute("file");
        if (!file) continue;
        tinyxml2::XMLDocument incDoc;
        const tinyxml2::XMLElement* incRoot = ResolveInclude(mj, ctx, file, incDoc);
        if (!incRoot) continue;
        if (const tinyxml2::XMLElement* incWorld = incRoot->FirstChildElement("worldbody")) {
            process_world(incWorld);
        }
    }
}

bool HasXmlExtension(const std::string& path)
{
    auto pos = path.find_last_of('.');
    if (pos == std::string::npos) return false;
    std::string ext = path.substr(pos);
    std::transform(ext.begin(), ext.end(), ext.begin(), ::tolower);
    return ext == ".xml";
}

}

RobotData RobotLoader::LoadRobot(const std::string& filepath, JPH::PhysicsSystem* physicsSystem)
{
    RobotData robot_data;

    if (physicsSystem == nullptr) {
        std::cerr << "RobotLoader: PhysicsSystem is null." << std::endl;
        return robot_data;
    }

    // --- MJCF path ---
    if (HasXmlExtension(filepath)) {
        std::filesystem::path xml_path = std::filesystem::absolute(filepath);
        tinyxml2::XMLDocument doc;
        if (doc.LoadFile(xml_path.string().c_str()) != tinyxml2::XML_SUCCESS) {
            std::cerr << "RobotLoader: Failed to load MJCF " << filepath << " error: " << doc.ErrorStr() << std::endl;
            return robot_data;
        }
        const tinyxml2::XMLElement* mj = doc.FirstChildElement("mujoco");
        if (!mj) {
            std::cerr << "RobotLoader: Not an MJCF file (missing <mujoco>)" << std::endl;
            return robot_data;
        }
        MJCFContext ctx;
        ctx.baseDir = xml_path.parent_path();
        if (const tinyxml2::XMLElement* comp = mj->FirstChildElement("compiler")) {
            const char* meshdir = comp->Attribute("meshdir");
            if (meshdir) ctx.meshDir = ctx.baseDir / meshdir;
        }
        ctx.defaults.Ingest(mj->FirstChildElement("default"));
        // Load mesh assets
        std::unordered_map<std::string, MeshData> meshCache;
        if (const tinyxml2::XMLElement* assets = mj->FirstChildElement("asset")) {
            for (const tinyxml2::XMLElement* m = assets->FirstChildElement("mesh"); m; m = m->NextSiblingElement("mesh")) {
                const char* name = m->Attribute("name");
                const char* file = m->Attribute("file");
                if (!name || !file) continue;
                std::filesystem::path meshPath = ctx.meshDir.empty() ? ctx.baseDir / file : ctx.meshDir / file;
                MeshData data;
                if (!LoadBinarySTL(meshPath, data.vertices, data.indices)) {
                    std::cerr << "RobotLoader: failed to load mesh file " << meshPath << std::endl;
                    continue;
                }
                meshCache[name] = std::move(data);
            }
        }
        JPH::BodyInterface& body_interface = physicsSystem->GetBodyInterface();
        const JPH::ObjectLayer dynamicLayer = Layers::MOVING_BASE;
        std::map<std::string, JPH::BodyID> body_map;
        ProcessWorlds(mj, body_interface, *physicsSystem, ctx, dynamicLayer, meshCache, body_map, robot_data);
        return robot_data;
    }

    std::ifstream file(filepath);
    if (!file.is_open()) {
        std::cerr << "RobotLoader: Failed to open " << filepath << std::endl;
        return robot_data;
    }

    nlohmann::json data;
    file >> data;

    std::map<std::string, JPH::BodyID> body_map;
    JPH::BodyInterface& body_interface = physicsSystem->GetBodyInterface();

    const auto& bodies = data.at("bodies");
    for (const auto& body_data : bodies) {
        const std::string name = body_data.value("name", "");
        const std::string shape = body_data.value("shape", "box");
        const std::string type = body_data.value("type", "dynamic");

        if (name.empty()) {
            std::cerr << "RobotLoader: Body missing name." << std::endl;
            continue;
        }

        const auto& position = body_data.at("position");
        const double px = position.at(0).get<double>();
        const double py = position.at(1).get<double>();
        const double pz = position.at(2).get<double>();

        JPH::Quat rotation = JPH::Quat::sIdentity();
        auto rotation_it = body_data.find("rotation");
        if (rotation_it != body_data.end()) {
            const auto& rot = *rotation_it;
            const float rx = JPH::DegreesToRadians(rot.at(0).get<float>());
            const float ry = JPH::DegreesToRadians(rot.at(1).get<float>());
            const float rz = JPH::DegreesToRadians(rot.at(2).get<float>());
            rotation = JPH::Quat::sEulerAngles(JPH::Vec3(rx, ry, rz));
        }

        const JPH::EMotionType motion_type = (type == "static") ? JPH::EMotionType::Static : JPH::EMotionType::Dynamic;
        const JPH::ObjectLayer layer = (motion_type == JPH::EMotionType::Static) ? Layers::STATIC : Layers::MOVING_BASE;

        JPH::ShapeSettings::ShapeResult shape_result;
        if (shape == "box") {
            const auto& half_extents = body_data.at("half_extents");
            const float hx = half_extents.at(0).get<float>();
            const float hy = half_extents.at(1).get<float>();
            const float hz = half_extents.at(2).get<float>();
            JPH::BoxShapeSettings shape_settings(JPH::Vec3(hx, hy, hz));
            shape_result = shape_settings.Create();
        } else if (shape == "sphere" || shape == "circle") {
            const float radius = body_data.at("radius").get<float>();
            JPH::SphereShapeSettings shape_settings(radius);
            shape_result = shape_settings.Create();
        } else {
            std::cerr << "RobotLoader: Unsupported shape for body '" << name << "': " << shape << std::endl;
            continue;
        }

        if (shape_result.HasError()) {
            std::cerr << "RobotLoader: Shape creation failed for body '" << name << "': " << shape_result.GetError() << std::endl;
            continue;
        }

        JPH::Ref<JPH::Shape> shape_ref = shape_result.Get();
        JPH::BodyCreationSettings body_settings(
            shape_ref,
            JPH::RVec3(px, py, pz),
            rotation,
            motion_type,
            layer
        );

        if (motion_type == JPH::EMotionType::Dynamic) {
            const float mass = body_data.value("mass", 1.0f);
            body_settings.mOverrideMassProperties = JPH::EOverrideMassProperties::CalculateInertia;
            body_settings.mMassPropertiesOverride.mMass = mass;

            body_settings.mFriction = 0.0f;
            body_settings.mRestitution = 1.0f;
            body_settings.mLinearDamping = 0.0f;
            body_settings.mAngularDamping = 0.0f;
        }

        JPH::Body* body = body_interface.CreateBody(body_settings);
        if (body == nullptr) {
            std::cerr << "RobotLoader: Failed to create body '" << name << "'." << std::endl;
            continue;
        }

        const JPH::BodyID body_id = body->GetID();
        body_interface.AddBody(body_id, motion_type == JPH::EMotionType::Dynamic ? JPH::EActivation::Activate : JPH::EActivation::DontActivate);

        body_map[name] = body_id;
        robot_data.bodies.push_back(body_id);
    }

    auto joints_it = data.find("joints");
    if (joints_it != data.end()) {
        for (const auto& joint_data : *joints_it) {
            const std::string type = joint_data.value("type", "");
            if (type != "hinge") {
                std::cerr << "RobotLoader: Unsupported joint type: " << type << std::endl;
                continue;
            }

            const std::string parent_name = joint_data.value("parent", "");
            const std::string child_name = joint_data.value("child", "");

            auto parent_it = body_map.find(parent_name);
            auto child_it = body_map.find(child_name);
            if (parent_it == body_map.end() || child_it == body_map.end()) {
                std::cerr << "RobotLoader: Joint references unknown bodies: " << parent_name << " -> " << child_name << std::endl;
                continue;
            }

            const auto& point = joint_data.at("point");
            const auto& axis = joint_data.at("axis");

            const double px = point.at(0).get<double>();
            const double py = point.at(1).get<double>();
            const double pz = point.at(2).get<double>();

            JPH::Vec3 hinge_axis(
                axis.at(0).get<float>(),
                axis.at(1).get<float>(),
                axis.at(2).get<float>()
            );

            if (hinge_axis.IsNearZero()) {
                hinge_axis = JPH::Vec3::sAxisY();
            } else {
                hinge_axis = hinge_axis.Normalized();
            }

            const float axis_dot_up = std::fabs(hinge_axis.GetY());
            JPH::Vec3 reference = axis_dot_up > 0.9f ? JPH::Vec3::sAxisX() : JPH::Vec3::sAxisY();
            JPH::Vec3 normal_axis = hinge_axis.Cross(reference);
            if (normal_axis.IsNearZero()) {
                normal_axis = hinge_axis.GetNormalizedPerpendicular();
            } else {
                normal_axis = normal_axis.Normalized();
            }

            JPH::HingeConstraintSettings settings;
            settings.mSpace = JPH::EConstraintSpace::WorldSpace;
            settings.mPoint1 = JPH::RVec3(px, py, pz);
            settings.mPoint2 = JPH::RVec3(px, py, pz);
            settings.mHingeAxis1 = hinge_axis;
            settings.mHingeAxis2 = hinge_axis;
            settings.mNormalAxis1 = normal_axis;
            settings.mNormalAxis2 = normal_axis;
            settings.mLimitsMin = -JPH::JPH_PI;
            settings.mLimitsMax = JPH::JPH_PI;
            settings.mLimitsSpringSettings.mFrequency = 0.0f;
            settings.mLimitsSpringSettings.mDamping = 0.0f;
            settings.mMaxFrictionTorque = 0.0f;

            const float motor_speed = joint_data.value("motor_speed", 0.0f);
            const float motor_torque = joint_data.value("motor_torque", 0.0f);
            if (motor_torque > 0.0f) {
                settings.mMotorSettings.mSpringSettings.mFrequency = 10.0f;
                settings.mMotorSettings.mSpringSettings.mDamping = 1.0f;
                settings.mMotorSettings.mMinTorqueLimit = -motor_torque;
                settings.mMotorSettings.mMaxTorqueLimit = motor_torque;
            }

            JPH::TwoBodyConstraint* constraint = body_interface.CreateConstraint(&settings, parent_it->second, child_it->second);
            if (constraint == nullptr) {
                std::cerr << "RobotLoader: Failed to create hinge constraint for joint." << std::endl;
                continue;
            }

            if (motor_torque > 0.0f) {
                JPH::HingeConstraint* hinge = static_cast<JPH::HingeConstraint*>(constraint);
                hinge->SetMotorState(JPH::EMotorState::Velocity);
                hinge->SetTargetAngularVelocity(0.0f);
            }

            physicsSystem->AddConstraint(constraint);
            robot_data.constraints.push_back(constraint);
        }
    }

    return robot_data;
}
