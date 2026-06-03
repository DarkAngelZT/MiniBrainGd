#pragma once

#include <godot_cpp/classes/object.hpp>
#include <godot_cpp/variant/array.hpp>
#include <godot_cpp/variant/packed_float32_array.hpp>
#include <godot_cpp/core/binder_common.hpp>
#include <godot_cpp/core/gdvirtual.gen.inc>

namespace godot {

    enum class AIAgentMode {
        TRAINING,
        INFERENCE
    };

class AIAgent: public Object {
    GDCLASS(AIAgent, Object);

protected:
    static void _bind_methods();
    AIAgentMode mode;

    //推理模式用这个
    MiniBrain::Network<MiniBrain::Scalar> *network = nullptr;
    //训练模式用这个
    MiniBrain::Network<MiniBrain::AutoDiffVar> *actor = nullptr;
    MiniBrain::Network<MiniBrain::AutoDiffVar> *critic = nullptr;
    MiniBrain::MatrixX<MiniBrain::AutoDiffVar> *m_training_input = nullptr;
    MiniBrain::MatrixX<MiniBrain::AutoDiffVar> *m_training_target = nullptr;
public:
    AIAgent(AIAgentMode mode);
    ~AIAgent();

    void Init();

    AIAgentMode get_mode() const;

    PackedFloat32Array ProcessSensorData(const PackedFloat32Array &data);

    void PushTrainingData(const godot::Array& batch_inputs, const godot::Array& batch_targets);

    void Train(int step);
};

} // namespace godot