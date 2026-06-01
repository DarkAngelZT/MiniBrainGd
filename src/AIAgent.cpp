#include "AIAgent.h"
#include <godot_cpp/core/class_db.hpp>
#include <godot_cpp/core/utility_functions.hpp>

using namespace godot;

void AIAgent::_bind_methods() {
    ClassDB::bind_method(D_METHOD("Init"), &AIAgent::Init);
    ClassDB::bind_method(D_METHOD("process_sensor_data", "data"), &AIAgent::ProcessSensorData);
    ClassDB::bind_method(D_METHOD("PushTrainingData"), &AIAgent::PushTrainingData);
    ClassDB::bind_method(D_METHOD("Train"), &AIAgent::Train);
}

AIAgent::AIAgent() {
    network = nullptr;
}

AIAgent::~AIAgent() {
    // network is owned/managed elsewhere for now.
    network = nullptr;
}

void AIAgent::Init() {
    // Placeholder initialization. Real network setup can be done later.
    UtilityFunctions::print("AIAgent: Init called");
}

PackedFloat32Array AIAgent::ProcessSensorData(const PackedFloat32Array &input) {
    PackedFloat32Array output;

    if (!network) {
        UtilityFunctions::push_error("AIAgent: network is not set");
        return output;
    }

    // For now, echo input back. Replace with real network inference later.
    output = input;
    return output;
}

void AIAgent::PushTrainingData() {
    UtilityFunctions::print("AIAgent: PushTrainingData called");
}

void AIAgent::Train() {
    UtilityFunctions::print("AIAgent: Train called");
}