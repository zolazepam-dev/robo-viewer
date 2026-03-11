#include "common.h"

class WebWorkerManager {
private:
    std::vector<std::thread> workers;
    std::atomic<bool> running{false};
    std::queue<std::function<void()>> taskQueue;
    std::mutex queueMutex;
    std::condition_variable cv;
    int numWorkers;
    
public:
    WebWorkerManager(int numWorkers = 8) : numWorkers(numWorkers) {}
    
    void start() {
        running = true;
        for (int i = 0; i < numWorkers; i++) {
            workers.emplace_back([this]() {
                workerLoop();
            });
        }
    }
    
    void stop() {
        running = false;
        cv.notify_all();
        for (auto& worker : workers) {
            if (worker.joinable()) worker.join();
        }
        workers.clear();
    }
    
    void enqueueTask(std::function<void()> task) {
        {
            std::lock_guard<std::mutex> lock(queueMutex);
            taskQueue.push(std::move(task));
        }
        cv.notify_one();
    }
    
private:
    void workerLoop() {
        while (running) {
            std::function<void()> task;
            {
                std::unique_lock<std::mutex> lock(queueMutex);
                cv.wait(lock, [this]() { return !taskQueue.empty() || !running; });
                
                if (!running && taskQueue.empty()) break;
                
                if (!taskQueue.empty()) {
                    task = std::move(taskQueue.front());
                    taskQueue.pop();
                }
            }
            
            if (task) {
                try {
                    task();
                } catch (...) {
                    // Handle task exception
                }
            }
        }
    }
};
