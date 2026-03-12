#include <GL/glew.h>
#include <GLFW/glfw3.h>
#include <iostream>
#include <vector>
#include <thread>
#include <atomic>
#include <chrono>
#include <mutex>
#include <filesystem>
#include <algorithm>
#include <cstring>
#include <omp.h>
#include <queue>
#include <condition_variable>

#include "src/VectorizedEnv.h"
#include "src/NeuralNetwork.h"
#include "src/TD3Trainer.h"
#include "src/Renderer.h"
#include "src/OverlayUI_refactor.h"
#include "src/ConfigManager.h"
#include "src/VisualState.h"
#include "Application.h"
#include "modules/common/PerformanceDiagnoser.h"

int main(int argc, char* argv[]) {
    // Parse command-line arguments
    int cmdNumEnvs = 0;
    int cmdRenderEnv = 0;
    for (int i = 1; i < argc; i++) {
        std::string arg = argv[i];
        if (arg == "--envs" && i + 1 < argc) cmdNumEnvs = std::stoi(argv[++i]);
        else if (arg == "--render-env" && i + 1 < argc) cmdRenderEnv = std::stoi(argv[++i]);
    }
    
    // Create application instance
    Application app;
    
    // Initialize application
    if (!app.Init(argc, argv)) {
        fprintf(stderr, "[main] Failed to initialize application\n");
        return -1;
    }
    
    // Run main loop
    app.Run();
    
    // Shutdown
    app.Shutdown();
    
    return 0;
}
