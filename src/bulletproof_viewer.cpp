#include <GL/glew.h>
#include <GLFW/glfw3.h>
#include <iostream>
#include <vector>
#include <string>
#include <regex>
#include <fstream>
#include <cmath>

struct Vertex { float x, y, z; };

std::vector<Vertex> vertices;

void ParseXML(const std::string& path) {
    std::ifstream file(path);
    std::string line;
    // Escaped correctly
    std::regex v_regex("<Vertex x=\"([^\"]+)\" y=\"([^\"]+)\" z=\"([^\"]+)\"");
    std::smatch match;
    while (std::getline(file, line)) {
        if (std::regex_search(line, match, v_regex)) {
            Vertex v;
            v.x = std::stof(match[1]);
            v.y = std::stof(match[2]);
            v.z = std::stof(match[3]);
            vertices.push_back(v);
        }
    }
}

int main(int argc, char** argv) {
    if (argc < 2) return 1;
    ParseXML(argv[1]);

    if (!glfwInit()) return 1;
    GLFWwindow* window = glfwCreateWindow(800, 800, "BULLETPROOF SPIKY VIEWER", nullptr, nullptr);
    if (!window) return 1;
    glfwMakeContextCurrent(window);
    
    float angle = 0.0f;
    while (!glfwWindowShouldClose(window)) {
        glfwPollEvents();
        glClearColor(0.0f, 0.0f, 0.05f, 1.0f); // Deep Space
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

        glMatrixMode(GL_PROJECTION);
        glLoadIdentity();
        // Wide view for massive spikes
        glOrtho(-20, 20, -20, 20, -100, 100);

        glMatrixMode(GL_MODELVIEW);
        glLoadIdentity();
        glRotatef(angle, 0.5f, 1.0f, 0.2f);
        angle += 0.3f; // Slower

        // Draw points
        glPointSize(8.0f);
        glBegin(GL_POINTS);
        glColor3f(1.0f, 0.9f, 0.0f); 
        for (const auto& v : vertices) glVertex3f(v.x, v.y, v.z);
        glEnd();

        // Draw structure
        glBegin(GL_LINES);
        glColor3f(0.0f, 0.7f, 1.0f);
        for (size_t i = 0; i < vertices.size(); ++i) {
            for (size_t j = i + 1; j < vertices.size(); ++j) {
                float dx = vertices[i].x - vertices[j].x;
                float dy = vertices[i].y - vertices[j].y;
                float dz = vertices[i].z - vertices[j].z;
                float dist = std::sqrt(dx*dx + dy*dy + dz*dz);
                // Connect skeleton AND spikes
                if (dist < 12.0f) { 
                    glVertex3f(vertices[i].x, vertices[i].y, vertices[i].z);
                    glVertex3f(vertices[j].x, vertices[j].y, vertices[j].z);
                }
            }
        }
        glEnd();

        glfwSwapBuffers(window);
    }
    glfwTerminate();
    return 0;
}
