// 3D Telemetry Grapher - Visual training metrics display
// Beautiful OpenGL visualization with ImGui

#include <Jolt/Jolt.h>
#include <GL/glew.h>
#include <GLFW/glfw3.h>

#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <string>
#include <chrono>
#include <cmath>
#include <algorithm>
#include <filesystem>

#include "imgui.h"
#include "imgui_impl_glfw.h"
#include "imgui_impl_opengl3.h"

struct TelemetryData {
    int step = 0;
    double sps = 0.0;
    double reward1 = 0.0;
    double reward2 = 0.0;
    double avgReward = 0.0;
    int bufferSize = 0;
    double timestamp = 0.0;
};

class TelemetryGrapher {
public:
    TelemetryGrapher(int width = 1600, int height = 1000) 
        : mWidth(width), mHeight(height)
    {
        if (!initGLFW()) {
            std::cerr << "Failed to initialize GLFW" << std::endl;
            exit(1);
        }
        
        initImGui();
        setupOpenGL();
    }
    
    ~TelemetryGrapher() {
        glDeleteVertexArrays(1, &mVAO);
        glDeleteBuffers(1, &mVBO);
        ImGui_ImplOpenGL3_Shutdown();
        ImGui_ImplGlfw_Shutdown();
        ImGui::DestroyContext();
        glfwDestroyWindow(mWindow);
        glfwTerminate();
    }
    
    void loadTelemetry(const std::string& filepath) {
        mData.clear();
        std::ifstream file(filepath);
        if (!file.is_open()) {
            std::cerr << "Cannot open telemetry file: " << filepath << std::endl;
            return;
        }
        
        std::string line;
        std::getline(file, line); // Skip header
        
        double startTime = -1.0;
        while (std::getline(file, line)) {
            std::istringstream iss(line);
            std::string token;
            std::vector<std::string> tokens;
            while (std::getline(iss, token, ',')) {
                tokens.push_back(token);
            }
            
            if (tokens.size() >= 4) {
                TelemetryData d;
                try {
                    d.step = std::stoi(tokens[0]);
                    std::string tag = tokens[1];
                    double val = std::stod(tokens[2]);
                    
                    if (tag == "SPS") d.sps = val;
                    else if (tag == "Reward1") d.reward1 = val;
                    else if (tag == "Reward2") d.reward2 = val;
                    else if (tag == "AvgReward") d.avgReward = val;
                    else if (tag == "Buffer_Size") d.bufferSize = val;
                    else continue;
                    
                    if (startTime < 0) startTime = d.step;
                    d.timestamp = (d.step - startTime) / 60.0;
                    
                    mData.push_back(d);
                } catch (...) {}
            }
        }
        
        std::cout << "[TelemetryGrapher] Loaded " << mData.size() << " data points" << std::endl;
    }
    
    void run() {
        auto lastRefresh = std::chrono::steady_clock::now();
        float cameraAngle = 0.0f;
        float cameraDistance = 12.0f;
        float cameraHeight = 3.0f;
        
        while (!glfwWindowShouldClose(mWindow)) {
            glfwPollEvents();
            
            // Start ImGui frame
            ImGui_ImplOpenGL3_NewFrame();
            ImGui_ImplGlfw_NewFrame();
            ImGui::NewFrame();
            
            // Auto-refresh every 2 seconds
            auto now = std::chrono::steady_clock::now();
            if (std::chrono::duration_cast<std::chrono::seconds>(now - lastRefresh).count() > 2) {
                if (mAutoRefresh && !mTelemetryPath.empty()) {
                    loadTelemetry(mTelemetryPath);
                }
                lastRefresh = now;
            }
            
            // Slow camera rotation
            if (mRotateCamera) {
                cameraAngle += 0.002f;
            }
            
            // ImGui docking
            ImGui::Begin("Main Window", nullptr, ImGuiWindowFlags_NoTitleBar | ImGuiWindowFlags_NoResize | ImGuiWindowFlags_NoMove);
            ImGui::SetWindowPos(ImVec2(0, 0));
            ImGui::SetWindowSize(ImGui::GetIO().DisplaySize);
            
            renderUI();
            ImGui::End(); // Main Window
            
            // Render 3D scene
            render3DGraphs(cameraAngle, cameraDistance, cameraHeight);
            
            // Render ImGui
            ImGui::Render();
            ImGui_ImplOpenGL3_RenderDrawData(ImGui::GetDrawData());
            
            glfwSwapBuffers(mWindow);
        }
    }
    
private:
    bool initGLFW() {
        if (!glfwInit()) return false;
        
        glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
        glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3);
        glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);
        glfwWindowHint(GLFW_SAMPLES, 4);
        
        mWindow = glfwCreateWindow(mWidth, mHeight, "JOLTrl 3D Telemetry Grapher", nullptr, nullptr);
        if (!mWindow) return false;
        
        glfwMakeContextCurrent(mWindow);
        glfwSwapInterval(1);
        
        if (glewInit() != GLEW_OK) return false;
        
        std::cout << "OpenGL: " << glGetString(GL_VERSION) << std::endl;
        return true;
    }
    
    void initImGui() {
        IMGUI_CHECKVERSION();
        ImGui::CreateContext();
        ImGuiIO& io = ImGui::GetIO();
        // io.ConfigFlags |= ImGuiConfigFlags_DockingEnable; // Not available
        io.ConfigFlags |= ImGuiConfigFlags_NavEnableKeyboard;
        
        ImGui::StyleColorsDark();
        ImGuiStyle& style = ImGui::GetStyle();
        style.WindowRounding = 10.0f;
        style.FrameRounding = 6.0f;
        style.GrabRounding = 6.0f;
        style.ScrollbarRounding = 6.0f;
        style.TabRounding = 6.0f;
        
        // Modern dark theme with gradients
        ImVec4* colors = style.Colors;
        colors[ImGuiCol_WindowBg] = ImVec4(0.04f, 0.04f, 0.06f, 0.95f);
        colors[ImGuiCol_ChildBg] = ImVec4(0.06f, 0.06f, 0.08f, 0.9f);
        colors[ImGuiCol_PopupBg] = ImVec4(0.08f, 0.08f, 0.12f, 0.95f);
        colors[ImGuiCol_Border] = ImVec4(0.15f, 0.15f, 0.2f, 0.5f);
        colors[ImGuiCol_Header] = ImVec4(0.15f, 0.35f, 0.55f, 1.0f);
        colors[ImGuiCol_HeaderHovered] = ImVec4(0.2f, 0.4f, 0.6f, 1.0f);
        colors[ImGuiCol_HeaderActive] = ImVec4(0.25f, 0.45f, 0.65f, 1.0f);
        colors[ImGuiCol_Button] = ImVec4(0.15f, 0.35f, 0.55f, 1.0f);
        colors[ImGuiCol_ButtonHovered] = ImVec4(0.2f, 0.4f, 0.6f, 1.0f);
        colors[ImGuiCol_ButtonActive] = ImVec4(0.25f, 0.45f, 0.65f, 1.0f);
        colors[ImGuiCol_Tab] = ImVec4(0.1f, 0.1f, 0.15f, 1.0f);
        colors[ImGuiCol_TabHovered] = ImVec4(0.2f, 0.2f, 0.3f, 1.0f);
        colors[ImGuiCol_TabActive] = ImVec4(0.15f, 0.2f, 0.3f, 1.0f);
        colors[ImGuiCol_PlotLines] = ImVec4(0.4f, 0.4f, 0.4f, 1.0f);
        colors[ImGuiCol_PlotLinesHovered] = ImVec4(0.7f, 0.4f, 0.0f, 1.0f);
        
        ImGui_ImplGlfw_InitForOpenGL(mWindow, true);
        ImGui_ImplOpenGL3_Init("#version 330");
    }
    
    void setupOpenGL() {
        glGenVertexArrays(1, &mVAO);
        glGenBuffers(1, &mVBO);
        
        glEnable(GL_DEPTH_TEST);
        glEnable(GL_BLEND);
        glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
        glEnable(GL_LINE_SMOOTH);
        glHint(GL_LINE_SMOOTH_HINT, GL_NICEST);
    }
    
    void render3DGraphs(float angle, float distance, float camHeight) {
        glClearColor(0.02f, 0.02f, 0.04f, 1.0f);
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
        
        int fbWidth, fbHeight;
        glfwGetFramebufferSize(mWindow, &fbWidth, &fbHeight);
        glViewport(0, 0, fbWidth, fbHeight);
        
        // Simple perspective projection
        float aspect = (float)fbWidth / (float)fbHeight;
        float fov = 45.0f;
        float near = 0.1f;
        float far = 100.0f;
        float f = 1.0f / tanf(fov * 3.14159f / 360.0f);
        
        mat4 proj = mat4(
            f/aspect, 0, 0, 0,
            0, f, 0, 0,
            0, 0, (far+near)/(near-far), -1,
            0, 0, (2*far*near)/(near-far), 0
        );
        
        // Camera view matrix
        float camX = cos(angle) * distance;
        float camZ = sin(angle) * distance;
        mat4 view = lookAt(vec3(camX, camHeight, camZ), vec3(0, 0, 0), vec3(0, 1, 0));
        
        mat4 vp = proj * view;
        
        if (mData.size() < 2) {
            // Draw "No Data" text placeholder
            return;
        }
        
        // Calculate data bounds
        float maxX = mData.size() - 1;
        float maxSPS = 10, maxReward = 2, maxBuffer = 10000;
        for (const auto& d : mData) {
            if (d.sps > maxSPS) maxSPS = d.sps;
            if (fabs(d.reward1) > maxReward) maxReward = fabs(d.reward1);
            if (fabs(d.reward2) > maxReward) maxReward = fabs(d.reward2);
            if (d.bufferSize > maxBuffer) maxBuffer = d.bufferSize;
        }
        maxSPS = std::max(maxSPS, 10.0f);
        maxReward = std::max(maxReward, 1.0f);
        maxBuffer = std::max(maxBuffer, 100.0f);
        
        // Draw graph base grid
        drawGrid(vp);
        
        // Draw 3D line graphs
        drawLineGraph(vp, "SPS", 0x00ffff, -3, 0, 
            [this, maxX, maxSPS](size_t i) {
                return vec3((float)i/maxX*6-3, mData[i].sps/maxSPS*3, 0);
            });
        
        drawLineGraph(vp, "Reward1", 0x00ff88, -3, 2,
            [this, maxX, maxReward](size_t i) {
                return vec3((float)i/maxX*6-3, (mData[i].reward1/maxReward+1)*1.5, 2);
            });
        
        drawLineGraph(vp, "Reward2", 0xffaa00, -3, 4,
            [this, maxX, maxReward](size_t i) {
                return vec3((float)i/maxX*6-3, (mData[i].reward2/maxReward+1)*1.5, 4);
            });
        
        drawLineGraph(vp, "AvgReward", 0xaa00ff, -3, -2,
            [this, maxX, maxReward](size_t i) {
                return vec3((float)i/maxX*6-3, (mData[i].avgReward/maxReward+1)*1.5, -2);
            });
        
        drawLineGraph(vp, "Buffer", 0xff4444, -3, -4,
            [this, maxX, maxBuffer](size_t i) {
                return vec3((float)i/maxX*6-3, mData[i].bufferSize/maxBuffer*3, -4);
            });
    }
    
    struct vec3 {
        float x, y, z;
        vec3(float _x, float _y, float _z) : x(_x), y(_y), z(_z) {}
    };
    
    struct mat4 {
        float m[16];
        mat4() { std::fill(m, m+16, 0); }
        mat4(float a,float b,float c,float d, float e,float f,float g,float h,
             float i,float j,float k,float l, float m1,float n,float o,float p) {
            m[0]=a; m[1]=b; m[2]=c; m[3]=d;
            m[4]=e; m[5]=f; m[6]=g; m[7]=h;
            m[8]=i; m[9]=j; m[10]=k; m[11]=l;
            m[12]=m1; m[13]=n; m[14]=o; m[15]=p;
        }
        mat4 operator*(const mat4& o) const {
            mat4 r;
            for(int i=0;i<4;i++) for(int j=0;j<4;j++)
                for(int k=0;k<4;k++) r.m[i*4+j] += m[i*4+k]*o.m[k*4+j];
            return r;
        }
    };
    
    mat4 lookAt(vec3 eye, vec3 center, vec3 up) {
        vec3 f(center.x-eye.x, center.y-eye.y, center.z-eye.z);
        float len = sqrtf(f.x*f.x+f.y*f.y+f.z*f.z);
        f.x/=len; f.y/=len; f.z/=len;
        
        vec3 r(f.y*up.z-f.z*up.y, f.z*up.x-f.x*up.z, f.x*up.y-f.y*up.x);
        len = sqrtf(r.x*r.x+r.y*r.y+r.z*r.z);
        r.x/=len; r.y/=len; r.z/=len;
        
        vec3 u(r.y*f.z-r.z*f.y, r.z*f.x-r.x*f.z, r.x*f.y-r.y*f.x);
        
        return mat4(
            r.x, r.y, r.z, -(r.x*eye.x+r.y*eye.y+r.z*eye.z),
            u.x, u.y, u.z, -(u.x*eye.x+u.y*eye.y+u.z*eye.z),
            -f.x, -f.y, -f.z, f.x*eye.x+f.y*eye.y+f.z*eye.z,
            0, 0, 0, 1
        );
    }
    
    void drawGrid(const mat4& vp) {
        glBindVertexArray(mVAO);
        glBindBuffer(GL_ARRAY_BUFFER, mVBO);
        
        std::vector<float> vertices;
        // Grid lines
        for (int i = -3; i <= 3; i++) {
            vertices.push_back(i); vertices.push_back(0); vertices.push_back(-4);
            vertices.push_back(i); vertices.push_back(0); vertices.push_back(4);
            vertices.push_back(-3); vertices.push_back(0); vertices.push_back(i);
            vertices.push_back(3); vertices.push_back(0); vertices.push_back(i);
        }
        
        glBufferData(GL_ARRAY_BUFFER, vertices.size()*sizeof(float), vertices.data(), GL_STATIC_DRAW);
        
        // Simple shader
        const char* vs = "#version 330\nlayout(location=0)in vec3 p;uniform mat4 u;void main(){gl_Position=u*vec4(p,1);}";
        const char* fs = "#version 330\nout vec4 c;void main(){c=vec4(0.15,0.15,0.25,0.5);}";
        
        GLuint prog = createShader(vs, fs);
        glUseProgram(prog);
        GLint loc = glGetUniformLocation(prog, "u");
        glUniformMatrix4fv(loc, 1, GL_FALSE, vp.m);
        
        glEnableVertexAttribArray(0);
        glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 0, 0);
        glDrawArrays(GL_LINES, 0, vertices.size()/3);
        glDisableVertexAttribArray(0);
        glDeleteProgram(prog);
        
        glBindBuffer(GL_ARRAY_BUFFER, 0);
        glBindVertexArray(0);
    }
    
    void drawLineGraph(const mat4& vp, const char* name, uint32_t color, float xOffset, float zOffset,
                       std::function<vec3(size_t)> getPoint) {
        if (mData.size() < 2) return;
        
        glBindVertexArray(mVAO);
        glBindBuffer(GL_ARRAY_BUFFER, mVBO);
        
        std::vector<float> vertices;
        for (size_t i = 0; i < mData.size(); i++) {
            vec3 p = getPoint(i);
            vertices.push_back(p.x);
            vertices.push_back(p.y);
            vertices.push_back(p.z);
        }
        
        glBufferData(GL_ARRAY_BUFFER, vertices.size()*sizeof(float), vertices.data(), GL_STATIC_DRAW);
        
        float r = ((color>>16)&0xff)/255.0f;
        float g = ((color>>8)&0xff)/255.0f;
        float b = (color&0xff)/255.0f;
        
        std::string fs = "#version 330\nout vec4 c;void main(){c=vec4(" + 
                         std::to_string(r) + "," + std::to_string(g) + "," + 
                         std::to_string(b) + ",0.9);}";
        
        const char* vs = "#version 330\nlayout(location=0)in vec3 p;uniform mat4 u;void main(){gl_Position=u*vec4(p,1);}";
        
        GLuint prog = createShader(vs, fs.c_str());
        glUseProgram(prog);
        GLint loc = glGetUniformLocation(prog, "u");
        glUniformMatrix4fv(loc, 1, GL_FALSE, vp.m);
        
        glLineWidth(3.0f);
        glEnableVertexAttribArray(0);
        glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 0, 0);
        glDrawArrays(GL_LINE_STRIP, 0, vertices.size()/3);
        glDisableVertexAttribArray(0);
        glDeleteProgram(prog);
        
        glBindBuffer(GL_ARRAY_BUFFER, 0);
        glBindVertexArray(0);
    }
    
    GLuint createShader(const char* vs, const char* fs) {
        GLuint prog = glCreateProgram();
        GLuint vsObj = glCreateShader(GL_VERTEX_SHADER);
        GLuint fsObj = glCreateShader(GL_FRAGMENT_SHADER);
        
        glShaderSource(vsObj, 1, &vs, nullptr);
        glShaderSource(fsObj, 1, &fs, nullptr);
        glCompileShader(vsObj);
        glCompileShader(fsObj);
        
        glAttachShader(prog, vsObj);
        glAttachShader(prog, fsObj);
        glLinkProgram(prog);
        
        glDeleteShader(vsObj);
        glDeleteShader(fsObj);
        
        return prog;
    }
    
    void renderUI() {
        // Stats window
        ImGui::Begin("📊 Training Statistics");
        
        if (!mData.empty()) {
            const auto& latest = mData.back();
            
            ImGui::Text("Steps: %d", latest.step);
            ImGui::SameLine();
            ImGui::TextColored(ImVec4(0.5,0.5,0.5,1), "|");
            ImGui::SameLine();
            ImGui::Text("Time: %.1f min", latest.timestamp / 60.0);
            
            ImGui::Separator();
            
            showStat("SPS", latest.sps, ImVec4(0, 1, 1, 1));
            showStat("Reward 1", latest.reward1, ImVec4(0, 1, 0.5, 1));
            showStat("Reward 2", latest.reward2, ImVec4(1, 0.67, 0, 1));
            showStat("Avg Reward", latest.avgReward, ImVec4(0.67, 0, 1, 1));
            showStat("Buffer", (double)latest.bufferSize, ImVec4(1, 0.27, 0.27, 1));
            
            ImGui::Separator();
            ImGui::Text("Data Points: %zu", mData.size());
        } else {
            ImGui::TextColored(ImVec4(1, 0.5, 0, 1), "⚠ No telemetry data loaded");
        }
        
        ImGui::End();
        
        // Controls window
        ImGui::Begin("🎮 Controls");
        
        ImGui::Text("Telemetry File:");
        ImGui::SetNextItemWidth(-1);
        ImGui::InputText("##telemetrypath", mTelemetryPathBuf, sizeof(mTelemetryPathBuf));
        
        if (ImGui::Button("Load File", ImVec2(-1, 35))) {
            mTelemetryPath = mTelemetryPathBuf;
            loadTelemetry(mTelemetryPath);
        }
        
        ImGui::Checkbox("🔄 Auto-refresh (2s)", &mAutoRefresh);
        ImGui::Checkbox("🎥 Rotate Camera", &mRotateCamera);
        
        ImGui::SliderFloat("Distance", &mCameraDistance, 5.0f, 20.0f);
        ImGui::SliderFloat("Height", &mCameraHeight, 1.0f, 10.0f);
        
        if (ImGui::Button("Reset View", ImVec2(-1, 30))) {
            mCameraDistance = 12.0f;
            mCameraHeight = 3.0f;
        }
        
        ImGui::Separator();
        ImGui::TextColored(ImVec4(0.5,0.5,0.5,1), "Mouse: Rotate | Scroll: Zoom");
        
        ImGui::End();
        
        // Legend window
        ImGui::Begin("📈 Legend");
        
        ImGui::ColorButton("##l_sps", ImVec4(0, 1, 1, 1), 0, ImVec2(25, 15));
        ImGui::SameLine(); ImGui::Text("SPS");
        
        ImGui::ColorButton("##l_r1", ImVec4(0, 1, 0.5, 1), 0, ImVec2(25, 15));
        ImGui::SameLine(); ImGui::Text("Reward 1");
        
        ImGui::ColorButton("##l_r2", ImVec4(1, 0.67, 0, 1), 0, ImVec2(25, 15));
        ImGui::SameLine(); ImGui::Text("Reward 2");
        
        ImGui::ColorButton("##l_avg", ImVec4(0.67, 0, 1, 1), 0, ImVec2(25, 15));
        ImGui::SameLine(); ImGui::Text("Avg Reward");
        
        ImGui::ColorButton("##l_buf", ImVec4(1, 0.27, 0.27, 1), 0, ImVec2(25, 15));
        ImGui::SameLine(); ImGui::Text("Buffer Size");
        
        ImGui::End();
    }
    
    void showStat(const char* label, double value, ImVec4 color) {
        std::string btnLabel = "##"; btnLabel += label;
        ImGui::ColorButton(btnLabel.c_str(), color, 0, ImVec2(15, 15));
        ImGui::SameLine();
        ImGui::Text("%s: %.3f", label, value);
    }
    
    GLFWwindow* mWindow = nullptr;
    int mWidth, mHeight;
    GLuint mVAO = 0, mVBO = 0;
    
    std::vector<TelemetryData> mData;
    std::string mTelemetryPath;
    char mTelemetryPathBuf[512] = "/home/cammyz/.joltrl/checkpoints/telemetry.csv";
    bool mAutoRefresh = true;
    bool mRotateCamera = true;
    float mCameraDistance = 12.0f;
    float mCameraHeight = 3.0f;
};

int main(int argc, char** argv) {
    std::string telemetryPath = "/home/cammyz/.joltrl/checkpoints/telemetry.csv";
    
    if (argc > 1) {
        telemetryPath = argv[1];
    }
    
    std::cout << "╔════════════════════════════════════════╗" << std::endl;
    std::cout << "║   JOLTrl 3D Telemetry Grapher          ║" << std::endl;
    std::cout << "╚════════════════════════════════════════╝" << std::endl;
    std::cout << "Telemetry: " << telemetryPath << std::endl;
    std::cout << "Controls:" << std::endl;
    std::cout << "  • Mouse drag: Rotate camera" << std::endl;
    std::cout << "  • Scroll: Zoom in/out" << std::endl;
    std::cout << "  • Auto-refreshes every 2 seconds" << std::endl;
    std::cout << "╔════════════════════════════════════════╗" << std::endl;
    
    TelemetryGrapher grapher(1600, 1000);
    grapher.loadTelemetry(telemetryPath);
    grapher.run();
    
    return 0;
}
