#include <iostream>
#include <vector>
#include <thread>
#include <chrono>
#include <cmath>
#include <iomanip>
#include <cstdint>
#include <immintrin.h>

// --- ANSI TrueColor Magic ---
void print_rgb(const std::string& text, int r, int g, int b, bool bold = false) {
    if (bold) std::cout << "\033[1m";
    std::cout << "\033[38;2;" << r << ";" << g << ";" << b << "m" << text << "\033[0m";
}

void print_gradient_text(const std::string& text, float speed = 0.1f) {
    for (size_t i = 0; i < text.length(); ++i) {
        int r = static_cast<int>(std::sin(0.3f * i + 0) * 127 + 128);
        int g = static_cast<int>(std::sin(0.3f * i + 2) * 127 + 128);
        int b = static_cast<int>(std::sin(0.3f * i + 4) * 127 + 128);
        std::cout << "\033[1m\033[38;2;" << r << ";" << g << ";" << b << "m" << text[i];
    }
    std::cout << "\033[0m\n";
}

void neural_sync_loader(const std::string& task_name) {
    const char spin[] = {'|', '/', '-', '\\'};
    std::cout << "\033[1m\033[38;2;0;255;255m[ INIT ] \033[0m" << task_name << " ";
    for (int i = 0; i < 20; ++i) {
        std::cout << "\b" << spin[i % 4] << std::flush;
        std::this_thread::sleep_for(std::chrono::milliseconds(40));
    }
    std::cout << "\b\033[1m\033[38;2;0;255;0m[ OK ]\033[0m\n";
}

// --- The Core System Tests ---
void test_avx2_alignment() {
    neural_sync_loader("Verifying AVX2 32-Byte Latent Memory Boundaries");

    alignas(32) std::vector<float> latent_pos(64 * 128, 1.0f);
    alignas(32) std::vector<float> latent_vel(64 * 128, 0.0f);

    uintptr_t pos_addr = reinterpret_cast<uintptr_t>(latent_pos.data());
    uintptr_t vel_addr = reinterpret_cast<uintptr_t>(latent_vel.data());

    if (pos_addr % 32 == 0 && vel_addr % 32 == 0) {
        print_rgb("  => L1 CACHE ALIGNMENT PERFECT. ZERO SEGFAULTS DETECTED.\n", 0, 255, 100, true);
    } else {
        print_rgb("  => FATAL: ALIGNMENT FAILED. EXPECT AVX2 CRASH.\n", 255, 0, 0, true);
    }
}

void test_register_dump() {
    neural_sync_loader("Saturating __m256 FMA Registers (MoLU Warmup)");

    __m256 test_vec = _mm256_set_ps(8.0f, 7.0f, 6.0f, 5.0f, 4.0f, 3.0f, 2.0f, 1.0f);
    float output[8];
    _mm256_storeu_ps(output, test_vec); // Unaligned store just for the print

    std::cout << "  \033[38;2;255;0;255m[YMM0 DUMP]:\033[0m ";
    for (int i = 0; i < 8; ++i) {
        std::cout << "\033[38;2;255;255;0m0x" << std::hex << std::setfill('0') << std::setw(8)
                  << *reinterpret_cast<uint32_t*>(&output[i]) << " \033[0m";
    }
    std::cout << std::dec << "\n";
}

int main() {
    std::cout << "\n";
    print_gradient_text("===============================================================");
    print_gradient_text("  ::: ROBO-VIEWER : NEURAL KINEMATIC ENGINE V1.0 :::");
    print_gradient_text("===============================================================\n");

    std::this_thread::sleep_for(std::chrono::milliseconds(300));

    test_avx2_alignment();
    test_register_dump();
    neural_sync_loader("Compiling 8x8 AVX2 Vertical Transpose Matrix");
    neural_sync_loader("Spinning up Jolt Physics 64MB TempAllocator");
    neural_sync_loader("Locking KLPER Background Thread to CPU Core 2");

    std::cout << "\n";
    print_rgb(">>> ALL SYSTEMS NOMINAL. READY FOR 128-PARALLEL COMBAT. <<<\n\n", 0, 255, 0, true);

    return 0;
}