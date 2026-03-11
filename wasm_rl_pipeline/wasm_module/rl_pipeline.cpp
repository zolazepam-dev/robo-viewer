#include "common.h"
#include "training_coordinator.cpp"

using namespace emscripten;

EMSCRIPTEN_BINDINGS(rl_pipeline) {
    class_<CombatTrainingCoordinator>("CombatTrainingCoordinator")
        .constructor<int>()
        .function("stepTraining", &CombatTrainingCoordinator::stepTraining, allow_raw_pointers())
        .function("resetTraining", &CombatTrainingCoordinator::resetTraining)
        .function("getObservations", &CombatTrainingCoordinator::getObservations, allow_raw_pointers())
        .function("getRewards", &CombatTrainingCoordinator::getRewards, allow_raw_pointers())
        .function("getDones", &CombatTrainingCoordinator::getDones, allow_raw_pointers())
        .function("getObservationDim", &CombatTrainingCoordinator::getObservationDim)
        .function("getActionDim", &CombatTrainingCoordinator::getActionDim)
        .function("getTotalEnvironments", &CombatTrainingCoordinator::getTotalEnvironments);
}

int main() {
    // This is called when the WASM module is loaded
    printf("JOLTrl WebAssembly Pipeline Initialized\n");
    return 0;
}
