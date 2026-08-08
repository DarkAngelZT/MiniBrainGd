#pragma once

#include <godot_cpp/classes/object.hpp>
#include <godot_cpp/variant/array.hpp>
#include <godot_cpp/variant/packed_float32_array.hpp>
#include <godot_cpp/core/binder_common.hpp>
#include <godot_cpp/core/gdvirtual.gen.inc>
#include <MNN/expr/ExprCreator.hpp>

#include <algorithm>
#include <memory>
#include <unordered_map>

#include "ActorNet.h"
#include "CriticNet.h"
#include "MovePolicy.h"

namespace MNN::Train { class ParameterOptimizer; }

namespace godot {

    struct TrainingData {
        MNN::Express::VARP player_state;
        MNN::Express::VARP monster_state;
        MNN::Express::VARP bullet_state;
        MNN::Express::VARP mask;
        MNN::Express::VARP actions;
        MNN::Express::VARP rewards;
        MNN::Express::VARP done;

        MNN::Express::VARP old_log_probs;
        MNN::Express::VARP old_critic_values; // 下一状态的Q值
        MNN::Express::VARP old_q_values; // 当前状态的Q值

        std::unordered_map<int, int> agent_write_index; // 智能体编号到当前写入位置
        int batch_size = 0;
        int num_frames = 1;
        int monster_entity_num = 0;
        int bullet_entity_num = 0;
        int player_dim = 0;
        int monster_dim = 0;
        int bullet_dim = 0;
        int action_dim = 0;

        MNN::Express::VARP buffer_player;
        MNN::Express::VARP buffer_monster;
        MNN::Express::VARP buffer_bullet;
        MNN::Express::VARP buffer_mask;
        MNN::Express::VARP buffer_action;
        MNN::Express::VARP buffer_log_probs;
        MNN::Express::VARP buffer_q_values;
        MNN::Express::VARP rollout_memory; // 用于存储每个智能体的GRU记忆，便于训练时恢复状态
        std::unordered_map<int, int> input_mapping; // 智能体编号到当前批次位置

        static void SetZero(const MNN::Express::VARP &tensor) {
            if (tensor == nullptr || tensor->getInfo() == nullptr) {
                return;
            }
            float *data = tensor->writeMap<float>();
            std::fill(data, data + tensor->getInfo()->size, 0.0f);
        }

        void Clear() {
            ClearTrainingData();
            SetZero(buffer_player);
            SetZero(buffer_monster);
            SetZero(buffer_bullet);
            SetZero(buffer_mask);
            SetZero(buffer_action);
            SetZero(buffer_q_values);
            SetZero(buffer_log_probs);
            input_mapping.clear();
            rollout_memory = nullptr;
        }

        void ClearTrainingData() {
            SetZero(player_state);
            SetZero(monster_state);
            SetZero(bullet_state);
            SetZero(mask);
            SetZero(actions);
            SetZero(rewards);
            SetZero(done);
            SetZero(old_log_probs);
            SetZero(old_critic_values);
            SetZero(old_q_values);
            agent_write_index.clear();
        }

        void Init(int inBatch_size, int inNum_frames,
                  int inMonsterEntityNum, int inBulletEntityNum,
                  int inPlayerDim, int inMonsterDim, int inBulletDim,
                  int inActionDim) {
            this->batch_size = inBatch_size;
            this->num_frames = inNum_frames;
            monster_entity_num = inMonsterEntityNum;
            bullet_entity_num = inBulletEntityNum;
            player_dim = inPlayerDim;
            monster_dim = inMonsterDim;
            bullet_dim = inBulletDim;
            action_dim = inActionDim;

            // 帧数据在首维连续存放，每一批取出后仍是[batch, entity, feature]。
            const int sample_capacity = inBatch_size * inNum_frames;
            player_state = MNN::Express::_Input({sample_capacity, 1, inPlayerDim}, MNN::Express::NCHW);
            monster_state = MNN::Express::_Input({sample_capacity, inMonsterEntityNum, inMonsterDim}, MNN::Express::NCHW);
            bullet_state = MNN::Express::_Input({sample_capacity, inBulletEntityNum, inBulletDim}, MNN::Express::NCHW);
            mask = MNN::Express::_Input({sample_capacity, 1 + inMonsterEntityNum + inBulletEntityNum}, MNN::Express::NCHW);
            actions = MNN::Express::_Input({sample_capacity, inActionDim}, MNN::Express::NCHW);
            rewards = MNN::Express::_Input({sample_capacity, 1}, MNN::Express::NCHW);
            done = MNN::Express::_Input({sample_capacity, 1}, MNN::Express::NCHW);
            old_critic_values = MNN::Express::_Input({sample_capacity, 1}, MNN::Express::NCHW);
            old_log_probs = MNN::Express::_Input({sample_capacity, 1}, MNN::Express::NCHW);
            old_q_values = MNN::Express::_Input({sample_capacity, 1}, MNN::Express::NCHW);

            buffer_player = MNN::Express::_Input({inBatch_size, 1, inPlayerDim}, MNN::Express::NCHW);
            buffer_monster = MNN::Express::_Input({inBatch_size, inMonsterEntityNum, inMonsterDim}, MNN::Express::NCHW);
            buffer_bullet = MNN::Express::_Input({inBatch_size, inBulletEntityNum, inBulletDim}, MNN::Express::NCHW);
            buffer_mask = MNN::Express::_Input({inBatch_size, 1 + inMonsterEntityNum + inBulletEntityNum}, MNN::Express::NCHW);
            buffer_action = MNN::Express::_Input({inBatch_size, inActionDim}, MNN::Express::NCHW);
            buffer_log_probs = MNN::Express::_Input({inBatch_size, 1}, MNN::Express::NCHW);
            buffer_q_values = MNN::Express::_Input({inBatch_size, 1}, MNN::Express::NCHW);
            Clear();
        }
    };

class AIAgent: public Object {
    GDCLASS(AIAgent, Object);
public:
    enum AIAgentMode {
        INFERENCE = 0,
        TRAINING = 1
    };
protected:
    static void _bind_methods();
    AIAgentMode mode;
    int m_insize,m_outSize;
    int m_monsterEntityNum = 0;
    int m_bulletEntityNum = 0;
    int m_playerDim = 0;
    int m_monsterDim = 0;
    int m_bulletDim = 0;

    // Actor 在推理和训练模式下共用，Critic 仅在训练模式下创建。
    MiniMind::ActorNet *m_mnnActorNet = nullptr;
    MiniMind::CriticNet *m_mnnCriticNet = nullptr;

    std::shared_ptr<TrainingData> m_training_data;
    std::shared_ptr<MNN::Train::ParameterOptimizer> m_actor_optimizer;
    std::shared_ptr<MNN::Train::ParameterOptimizer> m_critic_optimizer;

    float m_gamma = 0.93f;
    float m_lambda = 0.9f;
    float m_clip_epsilon = 0.2f;
    float m_continuous_gamma = 0.9f;
    float m_move_logistic_scale = MiniMind::MovePolicy::kInitialLogisticScale;

    MNN::Express::VARP CalculateLogProbs(
        const MNN::Express::VARP &actions,
        const MNN::Express::VARP &move_data,
        const MNN::Express::VARP &shoot_data,
        const MNN::Express::VARP &monster_mask,
        float move_logistic_scale);

    MNN::Express::VARP ComputeAdvantage(
        const MNN::Express::VARP &td_delta,
        const MNN::Express::VARP &done,
        float gamma, float lambda);

    void UpdateActorNetTrainingMode();
public:
    AIAgent();  // 无参构造函数供Godot使用
    AIAgent(AIAgentMode mode);
    ~AIAgent();

    void Init(
        int monster_entity_num, int bullet_entity_num,
        int player_dim, int monster_dim, int bullet_dim,
        int move_dim, int shoot_dim,
        int embedding_dim=16, int attention_key_dim=16, int gru_hidden_dim = 128,
        int out_hidden_dim = 128);

    AIAgentMode get_mode() const;
    void set_mode(AIAgentMode mode);

    PackedFloat32Array ProcessSensorData(
        const PackedFloat32Array &player, const PackedFloat32Array &monster,
        const PackedFloat32Array &bullet, bool isGameEnd = false);
    godot::Array BatchProcessSensorData(
        const godot::Array &batch_player, const godot::Array &batch_monster,
        const godot::Array &batch_bullet,
        const godot::PackedInt32Array &agent_ids);

    void PushTrainingData(const godot::PackedFloat32Array& batch_rewards, const godot::PackedInt32Array &agent_ids, const godot::PackedFloat32Array &batch_dones);

    void Train(int step);

    void SetBatchInfo(int batch_size, int action_dim, int num_frames=1);

    void SetLearningParameters(float gamma=0.93f, float lambda=0.9f, float clip_epsilon=0.2f, float continuous_gamma=0.9f);

    // 保存与加载网络参数
    void Save(const godot::String &parent_folder = godot::String("ai"), const godot::String &file_name = godot::String("checkpoint"));
    void Load(const godot::String &parent_folder = godot::String("ai"), const godot::String &file_name = godot::String("checkpoint"));


};

} // namespace godot

VARIANT_ENUM_CAST(AIAgent::AIAgentMode)
