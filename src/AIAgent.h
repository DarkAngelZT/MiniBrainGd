#pragma once

#include <godot_cpp/classes/object.hpp>
#include <godot_cpp/variant/packed_float32_array.hpp>
#include <godot_cpp/core/binder_common.hpp>
#include <godot_cpp/core/gdvirtual.gen.inc>

namespace godot {

class AIAgent: public Object {
    GDCLASS(AIAgent, Object);

protected:
    static void _bind_methods();

    // We keep network as an opaque pointer here; the agent checks for null
    // before using it. The neural network implementation is out-of-scope
    // for the binding layer and can be set/managed internally later.
    void *network = nullptr;

public:
    AIAgent();
    ~AIAgent();

    void Init();

    PackedFloat32Array ProcessSensorData(const PackedFloat32Array &data);

    void PushTrainingData();

    void Train();
};

} // namespace godot