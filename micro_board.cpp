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
    std::cout << "[DEBUG] Opening XML: " << path << std::endl;
    std::ifstream file(path);
    if (!file.is_open()) {
        std::cerr << "[ERROR] Could not open file: " << path << std::endl;
        return;
    }
    std::string line;
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
    std::cout << "[DEBUG] Loaded " << vertices.size() << " vertices." << std::endl;
}

int main(int argc, char** argv) {
    std::string xml = "dodecahedron.xml";
    if (argc > 1) xml = argv[1];
    
    ParseXML(xml);
    if (vertices.empty()) {
        std::cerr << "[ERROR] No geometry to render. Exiting." << std::endl;
        return 1;
    }

    std::cout << "[DEBUG] Initializing GLFW..." << std::endl;
    if (!glfwInit()) {
        std::cerr << "[ERROR] GLFW Initialization Failed" << std::endl;
        return 1;
    }

    std::cout << "[DEBUG] Creating Window..." << std::endl;
    GLFWwindow* window = glfwCreateWindow(800, 800, "DEBUG SPIKY POP", nullptr, nullptr);
    if (!window) {
        std::cerr << "[ERROR] Window creation failed. Is DISPLAY set correctly?" << std::endl;
        glfwTerminate();
        return 1;
    }
    glfwMakeContextCurrent(window);
    
    std::cout << "[DEBUG] Initializing GLEW..." << std::endl;
    glewExperimental = GL_TRUE;
    if (glewInit() != GLEW_OK) {
        std::cerr << "[ERROR] GLEW Initialization Failed" << std::endl;
        return 1;
    }

    std::cout << "[DEBUG] Starting Main Loop..." << std::endl;
    float angle = 0.0f;
    while (!glfwWindowShouldClose(window)) {
        glfwPollEvents();
        glClearColor(0.0f, 0.0f, 0.1f, 1.0f);
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
        
        glMatrixMode(GL_PROJECTION); glLoadIdentity(); glOrtho(-20, 20, -20, 20, -100, 100);
        glMatrixMode(GL_MODELVIEW); glLoadIdentity();
        glRotatef(angle, 0.5f, 1.0f, 0.2f); angle += 0.5f;
        
        glPointSize(8.0f); glBegin(GL_POINTS); glColor3f(1.0f, 0.9f, 0.0f);
        for (const auto& v : vertices) glVertex3f(v.x, v.y, v.z);
        glEnd();
        
        glBegin(GL_LINES); glColor3f(0.0f, 0.7f, 1.0f);
        for (size_t i = 0; i < vertices.size(); ++i) {
            for (size_t j = i + 1; j < vertices.size(); ++j) {
                float dx = vertices[i].x - vertices[j].x, dy = vertices[i].y - vertices[j].y, dz = vertices[i].z - vertices[j].z;
                if (std::sqrt(dx*dx + dy*dy + dz*dz) < 15.0f) {
                    glVertex3f(vertices[i].x, vertices[i].y, vertices[i].z);
                    glVertex3f(vertices[j].x, vertices[j].y, vertices[j].z);
                }
            }
        }
        glEnd();
        
        glfwSwapBuffers(window);
    }
    
    std::cout << "[DEBUG] Shutting down." << std::endl;
    glfwTerminate();
    return 0;
}
