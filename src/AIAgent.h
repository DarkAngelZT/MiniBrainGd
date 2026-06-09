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

    struct TrainingData {
        MiniBrain::Matrix<MiniBrain::Scalar> state;
        MiniBrain::Matrix<MiniBrain::Scalar> actions;
        MiniBrain::Matrix<MiniBrain::Scalar> rewards;
        MiniBrain::Matrix<MiniBrain::Scalar> done;

        MiniBrain::MatrixX<MiniBrain::Scalar> old_log_probs;
        MiniBrain::MatrixX<MiniBrain::Scalar> old_critic_values;

        void Clear() {
            state.setZero();
            actions.setZero();
            rewards.setZero();
            done.setZero();
            old_log_probs.setZero();
            old_critic_values.setZero();
        }
    };

class AIAgent: public Object {
    GDCLASS(AIAgent, Object);

protected:
    static void _bind_methods();
    AIAgentMode mode;
    int m_insize,m_outSize;

    //推理模式用这个
    MiniBrain::Network<MiniBrain::Scalar> *m_preprocessNet = nullptr;
    MiniBrain::Network<MiniBrain::Scalar> *m_moveNet = nullptr;
    MiniBrain::Network<MiniBrain::Scalar> *m_shootNet = nullptr;

    MiniBrain::Network<MiniBrain::Scalar>* m_GRULayer = nullptr;
    //训练模式用这个
    MiniBrain::Network<MiniBrain::AutoDiffVar> *m_actor_preprocessNet = nullptr;
    MiniBrain::Network<MiniBrain::AutoDiffVar> *m_actor_moveNet = nullptr;
    MiniBrain::Network<MiniBrain::AutoDiffVar> *m_actor_shootNet = nullptr;
    MiniBrain::Network<MiniBrain::Scalar>* m_actor_GRULayer = nullptr;

    MiniBrain::Network<MiniBrain::AutoDiffVar> *m_criticNet = nullptr;
    
    std::shared_ptr<TrainingData> m_training_data;
public:
    AIAgent(AIAgentMode mode);
    ~AIAgent();

    void Init(
        int input_dim, int output_dim, 
        int entity_feature_dim, int embedding_dim=16, int attention_key_dim=16, int gru_hidden_dim = 128,
        int out_hidden_dim = 128);

    AIAgentMode get_mode() const;

    PackedFloat32Array ProcessSensorData(const PackedFloat32Array &data);
    godot::Array BatchProcessSensorData(const godot::Array &batch_data);

    void PushTrainingData(const godot::Array& batch_inputs, const godot::Array& batch_targets);

    void Train(int step);
};

} // namespace godot