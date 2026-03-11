#include <emscripten.h>
extern "C" { EMSCRIPTEN_KEEPALIVE int test() { return 42; } }
