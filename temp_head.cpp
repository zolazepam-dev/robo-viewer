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
    return JPH::Vec3(x, y, z);
}

JPH::Quat ParseEuler(const char* str)
{
    if (!str) return JPH::Quat::sIdentity();
    float rx=0, ry=0, rz=0;
    std::stringstream ss(str);
    ss >> rx >> ry >> rz;
    if (ss.fail()) return JPH::Quat::sIdentity();
    return JPH::Quat::sEulerAngles(JPH::Vec3(JPH::DegreesToRadians(rx), JPH::DegreesToRadians(ry), JPH::DegreesToRadians(rz)));
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
            JPH::Vec3 p1(x1, y1, z1), p2(x2, y2, z2);
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
