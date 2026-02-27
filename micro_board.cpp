#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <map>
#include <string>
#include <iomanip>
#include <algorithm>
#include <cmath>

// Morphologica includes
#include <morph/Visual.h>
#include <morph/GraphVisual.h>
#include <morph/vvec.h>

struct Point {
    int step;
    float value;
};

class StandaloneBoard {
private:
    std::map<std::string, std::vector<Point>> data;

public:
    // Parse CSV line formatted as: Step,Tag,Value (e.g., 100,Loss,0.453)
    void parse_line(const std::string& line) {
        std::stringstream ss(line);
        std::string step_str, tag, value_str;

        if (std::getline(ss, step_str, ',') && 
            std::getline(ss, tag, ',') && 
            std::getline(ss, value_str, ',')) {
            try {
                int step = std::stoi(step_str);
                float value = std::stof(value_str);
                data[tag].push_back({step, value});
            } catch (...) {
                // Ignore malformed lines (e.g., headers)
            }
        }
    }

    void load_from_stream(std::istream& is) {
        std::string line;
        while (std::getline(is, line)) {
            parse_line(line);
        }
        
        // Sort data by step just in case lines arrived out of order
        for (auto& pair : data) {
            std::sort(pair.second.begin(), pair.second.end(), 
                [](const Point& a, const Point& b) { return a.step < b.step; });
        }
    }

    void render_gui() {
        if (data.empty()) return;

        // Initialize the morphologica visualization window in 3D Dark Mode
        morph::Visual v(1200, 800, "RL MicroBoard (3D Cascade Environment)");
        v.backgroundBlack(); // Switch to dark mode for better 3D contrast

        float z_offset = 0.0f; // Stack in depth (Z-axis) for 3D Ridgeline effect
        float x_offset = 0.0f; // Stagger horizontally
        float y_offset = 0.0f; // Stagger vertically

        for (const auto& pair : data) {
            const std::string& tag = pair.first;
            const auto& pts = pair.second;

            // Extract x and y into morph::vvec format
            morph::vvec<float> x_data, y_data;
            for (const auto& p : pts) {
                x_data.push_back(static_cast<float>(p.step));
                y_data.push_back(p.value);
            }

            // Create GraphVisual, offset its position in true 3D Space
            morph::vec<float, 3> offset = {x_offset, y_offset, z_offset};
            auto gv = std::make_unique<morph::GraphVisual<float>>(offset);
            v.bindmodel(gv);

            // Configure the graph aesthetics
            gv->setdata(x_data, y_data, tag);
            gv->xlabel = "Training Step";
            gv->ylabel = tag;

            gv->finalize();
            v.addVisualModel(gv); // Hand ownership to the window

            // Create the 3D cascade effect
            x_offset += 1.5f;   // Shift right slightly
            y_offset += 1.0f;   // Shift up slightly
            z_offset -= 3.0f;   // Push the next graph deeper into the screen!
        }

        std::cout << "\n[GUI] Launching 3D Morphologica environment.\n";
        std::cout << "      --> TIP: Click and drag with your mouse to rotate the 3D scene!\n";
        std::cout << "      --> TIP: Scroll your mouse wheel to fly through the cascade.\n";
        v.keepOpen(); // Blocks and runs the graphical render loop
    }

    // Generates a statistical breakdown specifically formatted for LLM context
    void generate_llm_breakdown(const std::string& tag) {
        if (data.find(tag) == data.end() || data[tag].empty()) return;

        const auto& pts = data[tag];
        float start_val = pts.front().value;
        float end_val = pts.back().value;
        float min_v = start_val, max_v = start_val;
        float sum = 0.0f;

        for (const auto& p : pts) {
            if (p.value < min_v) min_v = p.value;
            if (p.value > max_v) max_v = p.value;
            sum += p.value;
        }

        float avg = sum / pts.size();
        float delta_pct = ((end_val - start_val) / (std::abs(start_val) + 1e-8f)) * 100.0f;
        
        std::string trend = (end_val < start_val) ? "DECREASING" : "INCREASING";
        if (std::abs(delta_pct) < 1.0f) trend = "STABLE";

        std::cout << "=== LLM DATA BREAKDOWN FOR '" << tag << "' ===\n";
        std::cout << "Metric: " << tag << "\n";
        std::cout << "Data Points: " << pts.size() << "\n";
        std::cout << "Step Range: " << pts.front().step << " to " << pts.back().step << "\n";
        std::cout << "Start Value: " << std::fixed << std::setprecision(6) << start_val << "\n";
        std::cout << "End Value: " << end_val << "\n";
        std::cout << "Min Value: " << min_v << "\n";
        std::cout << "Max Value: " << max_v << "\n";
        std::cout << "Average: " << avg << "\n";
        std::cout << "Overall Trend: " << trend << " (" << std::showpos << delta_pct << "%)\n\n";
        
        std::cout << "[SYSTEM PROMPT APPEND]\n";
        std::cout << "The training metric '" << tag << "' exhibited an overall " << trend 
                  << " trend, changing by " << delta_pct << "%. "
                  << "The peak value was " << max_v << " and the minimum was " << min_v << ". "
                  << "Analyze these statistics to determine if the model is converging smoothly or if hyperparameter tuning (e.g., learning rate adjustment) is required.\n";
        std::cout << "=====================================\n\n";
    }

    void process_all() {
        if (data.empty()) {
            std::cout << "No valid log data found.\n";
            return;
        }

        // 1. Output the text/math logic to stdout for the LLM pipeline
        for (const auto& pair : data) {
            generate_llm_breakdown(pair.first);
        }

        // 2. Launch the hardware-accelerated GUI for the human viewing it
        render_gui();
    }
};

int main(int argc, char* argv[]) {
    StandaloneBoard board;

    if (argc > 1) {
        // Read from file if provided as argument
        std::ifstream file(argv[1]);
        if (!file.is_open()) {
            std::cerr << "Error: Could not open file " << argv[1] << "\n";
            return 1;
        }
        board.load_from_stream(file);
    } else {
        // Otherwise, read from standard input (pipe)
        board.load_from_stream(std::cin);
    }

    board.process_all();
    return 0;
}