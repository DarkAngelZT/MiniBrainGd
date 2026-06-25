#pragma once

#include <godot_cpp/classes/object.hpp>
#include <godot_cpp/variant/array.hpp>
#include <godot_cpp/variant/packed_float32_array.hpp>
#include <godot_cpp/core/binder_common.hpp>
#include <godot_cpp/core/gdvirtual.gen.inc>

#include <memory>
#include <unordered_map>

#include "../MiniBrain/Source/MiniBrain.h"

namespace godot {

    enum AIAgentMode {
        TRAINING,
        INFERENCE
    };

    struct TrainingData {
        MiniBrain::Matrix<MiniBrain::Scalar> state;
        MiniBrain::Matrix<MiniBrain::Scalar> actions;
        MiniBrain::Matrix<MiniBrain::Scalar> rewards;
        MiniBrain::Matrix<MiniBrain::Scalar> done;

        MiniBrain::Matrix<MiniBrain::Scalar> old_log_probs;
        MiniBrain::Matrix<MiniBrain::Scalar> old_critic_values;//q value for next state
        MiniBrain::Matrix<MiniBrain::Scalar> old_q_values;//q value of current state

        std::unordered_map<int, int> agent_write_index; // agent_id -> current write index
        int batch_size = 0;
        int num_frames = 1;
        int action_dim = 0;

        MiniBrain::Matrix<MiniBrain::Scalar> buffer_input;
        MiniBrain::Matrix<MiniBrain::Scalar> buffer_action;
        MiniBrain::Matrix<MiniBrain::Scalar> buffer_log_probs;
        MiniBrain::Matrix<MiniBrain::Scalar> buffer_q_values;
        std::unordered_map<int, int> input_mapping; // agent_id -> buffer column index        

        void Clear() {
            state.setZero();
            actions.setZero();
            rewards.setZero();
            done.setZero();
            old_log_probs.setZero();
            old_critic_values.setZero();
            old_q_values.setZero();
            buffer_input.setZero();
            buffer_action.setZero();
            buffer_q_values.setZero();
            buffer_log_probs.setZero();
            agent_write_index.clear();
            input_mapping.clear();
        }

        void Init(int inBatch_size, int inNum_frames, int state_dim, int action_dim) {
            this->batch_size = inBatch_size;
            this->num_frames = inNum_frames;
            this->action_dim = action_dim;
            state.resize(state_dim, inBatch_size*num_frames);
            actions.resize(action_dim, inBatch_size*num_frames);
            rewards.resize(1, inBatch_size*num_frames);
            done.resize(1, inBatch_size*num_frames);
            old_critic_values.resize(1, inBatch_size*num_frames);
            old_log_probs.resize(1, inBatch_size*num_frames);
            old_q_values.resize(1, inBatch_size*num_frames);

            buffer_input.resize(state_dim, inBatch_size);
            buffer_action.resize(action_dim, inBatch_size);
            buffer_log_probs.resize(1, inBatch_size);
            buffer_q_values.resize(1, inBatch_size);
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

    MiniBrain::GRU<MiniBrain::Scalar>* m_GRULayer = nullptr;
    //训练模式用这个
    MiniBrain::Network<MiniBrain::AutoDiffVar> *m_actor_preprocessNet = nullptr;
    MiniBrain::Network<MiniBrain::AutoDiffVar> *m_actor_moveNet = nullptr;
    MiniBrain::Network<MiniBrain::AutoDiffVar> *m_actor_shootNet = nullptr;
    MiniBrain::GRU<MiniBrain::AutoDiffVar>* m_actor_GRULayer = nullptr;

    MiniBrain::Network<MiniBrain::AutoDiffVar> *m_criticNet = nullptr;
    
    std::shared_ptr<TrainingData> m_training_data;
    std::shared_ptr<MiniBrain::Adam> m_optimizer;

    float m_gamma = 0.93f;
    float m_lambda = 0.9f;
    float m_clip_epsilon = 0.2f;
    float m_continuous_gamma = 0.9f;

    void CalculateLogProbs(
        const MiniBrain::Matrix<MiniBrain::Scalar> &action_new, 
        const MiniBrain::Matrix<MiniBrain::AutoDiffVar> &moveData, 
        const MiniBrain::Matrix<MiniBrain::AutoDiffVar> &shootData,
        MiniBrain::Matrix<MiniBrain::AutoDiffVar> &log_probs);

    void ComputeAdvantage(
        const MiniBrain::Matrix<MiniBrain::Scalar> &inTdDelta,
        float gamma, float lambda,
        MiniBrain::Matrix<MiniBrain::Scalar>& outAdvantage);
public:
    AIAgent();  // 无参构造函数供Godot使用
    AIAgent(AIAgentMode mode);
    ~AIAgent();

    void Init(
        int input_dim, int move_dim, int shoot_dim,
        int entity_feature_dim, int embedding_dim=16, int attention_key_dim=16, int gru_hidden_dim = 128,
        int out_hidden_dim = 128);

    AIAgentMode get_mode() const;
    void set_mode(AIAgentMode mode);

    PackedFloat32Array ProcessSensorData(const PackedFloat32Array &data);
    godot::Array BatchProcessSensorData(const godot::Array &batch_data, const godot::PackedInt32Array &agent_ids);

    void PushTrainingData(const godot::PackedFloat32Array& batch_rewards, const godot::PackedInt32Array &agent_ids, const godot::PackedFloat32Array &batch_dones);

    void Train(int step);

    void SetBatchInfo(int batch_size, int action_dim, int num_frames=1);

    void SetLearningParameters(float gamma=0.93f, float lambda=0.9f, float clip_epsilon=0.2f, float continuous_gamma=0.9f);
};

} // namespace godot

VARIANT_ENUM_CAST(godot::AIAgentMode)