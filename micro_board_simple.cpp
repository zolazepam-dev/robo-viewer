#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <map>
#include <string>
#include <iomanip>
#include <algorithm>
#include <cmath>

struct Point {
    int step;
    float value;
};

class StandaloneBoard {
private:
    std::map<std::string, std::vector<Point>> data;

public:
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
        
        for (auto& pair : data) {
            std::sort(pair.second.begin(), pair.second.end(), 
                [](const Point& a, const Point& b) { return a.step < b.step; });
        }
    }

    void render_ascii_graph(const std::string& tag, const std::vector<Point>& pts) {
        const int width = 60;
        const int height = 10;
        
        if (pts.empty()) return;
        
        float min_val = pts[0].value, max_val = pts[0].value;
        for (const auto& p : pts) {
            min_val = std::min(min_val, p.value);
            max_val = std::max(max_val, p.value);
        }
        
        float range = max_val - min_val;
        if (range < 1e-6) range = 1.0f;
        
        std::cout << "\n" << tag << "\n";
        std::cout << std::string(width + 10, '-') << "\n";
        std::cout << std::fixed << std::setprecision(4);
        std::cout << "Max: " << max_val << "\n";
        
        // Simple ASCII graph
        for (int row = height; row >= 0; --row) {
            float row_val = min_val + (range * row / height);
            std::cout << std::setw(8) << row_val << " |";
            
            for (size_t i = 0; i < pts.size() && i < (size_t)width; ++i) {
                int col = (int)((pts[i].value - min_val) / range * height);
                if (col == row) std::cout << "*";
                else std::cout << " ";
            }
            std::cout << "\n";
        }
        
        std::cout << "Min: " << min_val << "\n";
        std::cout << "Data points: " << pts.size() << "\n";
    }

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

        std::cout << "\n=== LLM DATA BREAKDOWN FOR '" << tag << "' ===\n";
        std::cout << "Metric: " << tag << "\n";
        std::cout << "Data Points: " << pts.size() << "\n";
        std::cout << "Step Range: " << pts.front().step << " to " << pts.back().step << "\n";
        std::cout << "Start Value: " << std::fixed << std::setprecision(6) << start_val << "\n";
        std::cout << "End Value: " << end_val << "\n";
        std::cout << "Min Value: " << min_v << "\n";
        std::cout << "Max Value: " << max_v << "\n";
        std::cout << "Average: " << avg << "\n";
        std::cout << "Overall Trend: " << trend << " (" << std::showpos << delta_pct << "%)\n";
    }

    void process_all() {
        if (data.empty()) {
            std::cout << "No valid log data found.\n";
            return;
        }

        for (const auto& pair : data) {
            generate_llm_breakdown(pair.first);
            render_ascii_graph(pair.first, pair.second);
        }
    }
};

int main(int argc, char* argv[]) {
    StandaloneBoard board;

    if (argc > 1) {
        std::ifstream file(argv[1]);
        if (!file.is_open()) {
            std::cerr << "Error: Could not open file " << argv[1] << "\n";
            return 1;
        }
        board.load_from_stream(file);
    } else {
        board.load_from_stream(std::cin);
    }

    board.process_all();
    return 0;
}
