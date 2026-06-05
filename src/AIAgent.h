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
    MiniBrain::Network<MiniBrain::Scalar> *m_preprocessNet = nullptr;
    MiniBrain::Network<MiniBrain::Scalar> *m_moveNet = nullptr;
    MiniBrain::Network<MiniBrain::Scalar> *m_shootNet = nullptr;
    //训练模式用这个
    MiniBrain::Network<MiniBrain::AutoDiffVar> *m_actor_preprocessNet = nullptr;
    MiniBrain::Network<MiniBrain::AutoDiffVar> *m_actor_moveNet = nullptr;
    MiniBrain::Network<MiniBrain::AutoDiffVar> *m_actor_shootNet = nullptr;

    MiniBrain::Network<MiniBrain::AutoDiffVar> *m_criticNet = nullptr;
    
    MiniBrain::MatrixX<MiniBrain::AutoDiffVar> *m_training_input = nullptr;
    MiniBrain::MatrixX<MiniBrain::AutoDiffVar> *m_training_target = nullptr;
public:
    AIAgent(AIAgentMode mode);
    ~AIAgent();

    void Init(int input_dim, int hidden_dim, int output_dim);

    AIAgentMode get_mode() const;

    PackedFloat32Array ProcessSensorData(const PackedFloat32Array &data);

    void PushTrainingData(const godot::Array& batch_inputs, const godot::Array& batch_targets);

    void Train(int step);
};

} // namespace godot