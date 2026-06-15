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

        MiniBrain::Matrix<MiniBrain::Scalar> old_log_probs;
        MiniBrain::Matrix<MiniBrain::Scalar> old_critic_values;

        std::unordered_map<int, int> agent_write_index; // agent_id -> current write index
        int batch_size = 0;
        int num_frames = 1;

        MiniBrain::Matrix<MiniBrain::Scalar> input_buffer;
        std::unordered_map<int, int> input_mapping; // agent_id -> buffer column index        

        void Clear() {
            state.setZero();
            actions.setZero();
            rewards.setZero();
            done.setZero();
            old_log_probs.setZero();
            old_critic_values.setZero();

            input_buffer.setZero();
            agent_write_index.clear();
            input_mapping.clear();
        }

        void Init(int inBatch_size, int inNum_frames, int state_dim, int action_dim) {
            this->batch_size = inBatch_size;
            this->num_frames = inNum_frames;
            state.resize(state_dim, inBatch_size*num_frames);
            actions.resize(action_dim, inBatch_size*num_frames);
            rewards.resize(1, inBatch_size*num_frames);
            done.resize(1, inBatch_size*num_frames);
            old_log_probs.resize(1, inBatch_size*num_frames);
            old_critic_values.resize(1, inBatch_size*num_frames);

            input_buffer.resize(state_dim, inBatch_size);
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
    godot::Array BatchProcessSensorData(const godot::Array &batch_data, const godot::Array &agent_ids);

    void PushTrainingData(const godot::Array& batch_rewards, const godot::Array& agent_ids, const godot::Array& batch_dones);

    void Train(int step);

    void SetBatchInfo(int batch_size, int num_frames=1);
};

} // namespace godot