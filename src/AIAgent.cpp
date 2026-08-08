#include "AIAgent.h"
#include "EntityMask.h"
#include "MovePolicy.h"
#include "ShootPolicy.h"
#include <MNN/expr/ExecutorScope.hpp>
#include <MNN/expr/MathOp.hpp>
#include <MNN/expr/NeuralNetWorkOp.hpp>
#include "../MiniMind/src/3rd_party/MNN/tools/train/source/optimizer/ParameterOptimizer.hpp"
#include <godot_cpp/core/class_db.hpp>
#include <godot_cpp/classes/dir_access.hpp>
#include <godot_cpp/classes/file_access.hpp>
#include <godot_cpp/classes/project_settings.hpp>
#include <godot_cpp/variant/utility_functions.hpp>

#include <algorithm>
#include <cmath>
#include <random>
#include <unordered_set>
#include <vector>

using namespace godot;

void AIAgent::_bind_methods() 
{
    ClassDB::bind_method(D_METHOD("Init", "monster_entity_num", "bullet_entity_num", "player_dim", "monster_dim", "bullet_dim", "move_dim", "shoot_dim", "embedding_dim", "attention_key_dim", "gru_hidden_dim", "out_hidden_dim"), &AIAgent::Init, DEFVAL(16), DEFVAL(16), DEFVAL(128), DEFVAL(128));
    ClassDB::bind_method(D_METHOD("get_mode"), &AIAgent::get_mode);
    ClassDB::bind_method(D_METHOD("set_mode", "mode"), &AIAgent::set_mode);
    ClassDB::bind_method(D_METHOD("ProcessSensorData", "player", "monster", "bullet", "is_game_end"), &AIAgent::ProcessSensorData, DEFVAL(false));
    ClassDB::bind_method(D_METHOD("BatchProcessSensorData", "batch_player", "batch_monster", "batch_bullet", "agent_ids"), &AIAgent::BatchProcessSensorData);
    ClassDB::bind_method(D_METHOD("PushTrainingData", "batch_rewards", "agent_ids", "batch_dones"), &AIAgent::PushTrainingData);
    ClassDB::bind_method(D_METHOD("Train", "step"), &AIAgent::Train);
    ClassDB::bind_method(D_METHOD("SetBatchInfo", "batch_size", "action_dim", "num_frames"), &AIAgent::SetBatchInfo, DEFVAL(1));
    ClassDB::bind_method(D_METHOD("SetLearningParameters", "gamma", "lambda", "clip_epsilon", "continuous_gamma"), &AIAgent::SetLearningParameters, DEFVAL(0.93f), DEFVAL(0.9f), DEFVAL(0.2f), DEFVAL(0.9f));

    ClassDB::bind_method(D_METHOD("Save", "parent_folder", "file_name"), &AIAgent::Save,
        DEFVAL("ai"), DEFVAL("checkpoint"));
    ClassDB::bind_method(D_METHOD("Load", "parent_folder", "file_name"), &AIAgent::Load,
        DEFVAL("ai"), DEFVAL("checkpoint"));

    BIND_ENUM_CONSTANT(TRAINING);
    BIND_ENUM_CONSTANT(INFERENCE);
}

namespace {
String Utf8(const char *text)
{
    return String::utf8(text);
}

MNN::Express::VARP Slice2D(
    const MNN::Express::VARP &value,
    int row_start, int column_start, int row_count, int column_count)
{
    using namespace MNN::Express;
    const int starts[] = {row_start, column_start};
    const int sizes[] = {row_count, column_count};
    return _Slice(
        value,
        _Const(starts, {2}, NCHW, halide_type_of<int>()),
        _Const(sizes, {2}, NCHW, halide_type_of<int>()));
}
bool CopyFlattenedState(
    const PackedFloat32Array &sample,
    int entity_num,
    int feature_dim,
    float *destination)
{
    if (entity_num <= 0 || feature_dim <= 0 || destination == nullptr ||
        sample.size() != entity_num * feature_dim) {
        return false;
    }

    // Godot 按 entity 依次横向拼接 feature；MNN 的 [B,E,F] 最后一维也连续，
    // 因而这里只需逐实体连续复制并赋予三维形状，不需要转置。
    const float *source = sample.ptr();
    for (int entity = 0; entity < entity_num; ++entity) {
        const int entity_offset = entity * feature_dim;
        std::copy_n(source + entity_offset, feature_dim,
                    destination + entity_offset);
    }
    return true;
}

MNN::Express::VARP log_sigmoid(const MNN::Express::VARP &x) {
    using namespace MNN::Express;
    // log(sigmoid(x)) = min(x, 0) - log1p(exp(-abs(x))).
    // This form keeps the exponential argument non-positive, so large logits do
    // not overflow. MNN's VARP has no unary '-' operator; _Negative builds the
    // corresponding differentiable Express node explicitly.
    const auto zero = _Scalar<float>(0.0f);
    return _Minimum(x, zero) - _Log1p(_Exp(_Negative(_Abs(x))));
}

}

MNN::Express::VARP godot::AIAgent::CalculateLogProbs(
    const MNN::Express::VARP &actions,
    const MNN::Express::VARP &move_data,
    const MNN::Express::VARP &shoot_data,
    const MNN::Express::VARP &monster_mask,
    float move_logistic_scale)
{
    if (actions == nullptr || move_data == nullptr || shoot_data == nullptr ||
        monster_mask == nullptr) {
        UtilityFunctions::push_error(Utf8("CalculateLogProbs????????"));
        return nullptr;
    }
    const auto *action_info = actions->getInfo();
    const auto *move_info = move_data->getInfo();
    const auto *shoot_info = shoot_data->getInfo();
    const auto *mask_info = monster_mask->getInfo();
    if (action_info == nullptr || move_info == nullptr || shoot_info == nullptr ||
        mask_info == nullptr || action_info->dim.size() != 2 ||
        move_info->dim.size() != 2 || shoot_info->dim.size() != 2 ||
        mask_info->dim.size() != 2 || action_info->dim[0] != move_info->dim[0] ||
        action_info->dim[0] != shoot_info->dim[0] ||
        action_info->dim[0] != mask_info->dim[0] || action_info->dim[1] != 4 ||
        move_info->dim[1] != 2 || mask_info->dim[1] != m_monsterEntityNum ||
        shoot_info->dim[1] != m_monsterEntityNum + 1) {
        UtilityFunctions::push_error(Utf8("CalculateLogProbs?????????"));
        return nullptr;
    }
    const auto move_log_probability = MiniMind::MovePolicy::LogisticLogProbability(
        actions, move_data, move_logistic_scale);
    const auto shoot_log_probability = MiniMind::ShootPolicy::LogProbability(
        actions, shoot_data, monster_mask);
    return move_log_probability + shoot_log_probability;
}

MNN::Express::VARP godot::AIAgent::ComputeAdvantage(
    const MNN::Express::VARP &td_delta,
    const MNN::Express::VARP &done,
    float gamma, float lambda)
{
    if (td_delta == nullptr || done == nullptr) {
        UtilityFunctions::push_error(Utf8("ComputeAdvantage：输入不能为空。"));
        return nullptr;
    }
    const auto *delta_info = td_delta->getInfo();
    const auto *done_info = done->getInfo();
    if (delta_info == nullptr || done_info == nullptr ||
        delta_info->dim != done_info->dim || delta_info->dim.size() != 2 ||
        delta_info->dim[1] != 1) {
        UtilityFunctions::push_error(Utf8("ComputeAdvantage：输入必须为单个智能体的[frames,1]。"));
        return nullptr;
    }

    const int frame_count = delta_info->dim[0];
    const float *delta_values = td_delta->readMap<float>();
    const float *done_values = done->readMap<float>();
    std::vector<float> advantages(static_cast<std::size_t>(frame_count));
    float last_advantage = 0.0f;
    for (int frame = frame_count - 1; frame >= 0; --frame) {
        last_advantage = delta_values[frame]
            + gamma * lambda * (1.0f - done_values[frame]) * last_advantage;
        advantages[frame] = last_advantage;
    }
    return MNN::Express::_Const(
        advantages.data(), {frame_count, 1}, MNN::Express::NCHW);
}

AIAgent::AIAgent() : mode(AIAgentMode::INFERENCE)
{
}

AIAgent::AIAgent(AIAgentMode mode) : mode(mode)
{
}

AIAgent::~AIAgent()
{
    m_actor_optimizer.reset();
    m_critic_optimizer.reset();
    m_insize = 0;
    m_outSize = 0;
    m_monsterEntityNum = m_bulletEntityNum = 0;
    m_playerDim = m_monsterDim = m_bulletDim = 0;
    delete m_mnnActorNet;
    delete m_mnnCriticNet;
}

godot::AIAgent::AIAgentMode AIAgent::get_mode() const {
    return mode;
}

void godot::AIAgent::set_mode(godot::AIAgent::AIAgentMode inMode)
{
    if (inMode != AIAgentMode::INFERENCE && inMode != AIAgentMode::TRAINING) {
        UtilityFunctions::push_error(Utf8("AIAgent.set_mode：mode取值无效。"));
        return;
    }
    mode = inMode;
    UpdateActorNetTrainingMode();
}

void AIAgent::UpdateActorNetTrainingMode()
{
    const bool is_training = mode == AIAgentMode::TRAINING;
    if (m_mnnActorNet != nullptr) {
        m_mnnActorNet->setIsTraining(is_training);
    }
    MNN::Express::ExecutorScope::Current()->lazyEval = is_training;
}

void AIAgent::Init(
    int monster_entity_num, int bullet_entity_num,
    int player_dim, int monster_dim, int bullet_dim,
    int move_dim, int shoot_dim,
    int embedding_dim,
    int attention_key_dim, int gru_hidden_dim,
    int out_hidden_dim)
{
    delete m_mnnActorNet;
    delete m_mnnCriticNet;
    m_mnnActorNet = nullptr;
    m_mnnCriticNet = nullptr;
    m_actor_optimizer.reset();
    m_critic_optimizer.reset();
    m_training_data.reset();
    m_insize = 0;
    m_outSize = 0;
    m_monsterEntityNum = m_bulletEntityNum = 0;
    m_playerDim = m_monsterDim = m_bulletDim = 0;
    m_move_logistic_scale = MiniMind::MovePolicy::kInitialLogisticScale;

    if (monster_entity_num <= 0 || bullet_entity_num <= 0 ||
        player_dim <= 0 || monster_dim <= 0 || bullet_dim <= 0 || move_dim != 2 ||
        shoot_dim != monster_entity_num + 1 || embedding_dim <= 0 || attention_key_dim <= 0 ||
        gru_hidden_dim <= 0 || out_hidden_dim <= 0) {
        UtilityFunctions::push_error(
            Utf8("AIAgent.Init：move_dim必须为2，shoot_dim必须等于怪物数量加1，其他维度必须为正数。"));
        return;
    }

    m_insize = player_dim + monster_entity_num * monster_dim +
        bullet_entity_num * bullet_dim;
    m_outSize = move_dim + shoot_dim;
    m_monsterEntityNum = monster_entity_num;
    m_bulletEntityNum = bullet_entity_num;
    m_playerDim = player_dim;
    m_monsterDim = monster_dim;
    m_bulletDim = bullet_dim;

    m_mnnActorNet = new MiniMind::ActorNet(
        player_dim, monster_dim, bullet_dim, monster_entity_num, move_dim, embedding_dim,
        attention_key_dim, gru_hidden_dim, out_hidden_dim);

    if (mode == AIAgentMode::TRAINING) {
        m_mnnCriticNet = new MiniMind::CriticNet(gru_hidden_dim, gru_hidden_dim * 3);
        m_training_data = std::make_shared<TrainingData>();

        auto actor_module = std::shared_ptr<MNN::Express::Module>(
            m_mnnActorNet, [](MNN::Express::Module*) {});
        auto critic_module = std::shared_ptr<MNN::Express::Module>(
            m_mnnCriticNet, [](MNN::Express::Module*) {});
        m_actor_optimizer.reset(MNN::Train::ParameterOptimizer::createADAM(
            actor_module, 0.001f, 0.9f, 0.999f, 0.0f, 1.0e-8f,
            MNN::Train::ParameterOptimizer::L2));
        m_critic_optimizer.reset(MNN::Train::ParameterOptimizer::createADAM(
            critic_module, 0.001f, 0.9f, 0.999f, 0.0f, 1.0e-8f,
            MNN::Train::ParameterOptimizer::L2));
    } else {
        m_training_data.reset();
    }

    UpdateActorNetTrainingMode();
}

PackedFloat32Array AIAgent::ProcessSensorData(
    const PackedFloat32Array &player_input,
    const PackedFloat32Array &monster_input,
    const PackedFloat32Array &bullet_input,
    bool isGameEnd)
{
    PackedFloat32Array output;
    if (m_mnnActorNet == nullptr) {
        UtilityFunctions::push_error("AIAgent.ProcessSensorData: Actor is not initialized.");
        return output;
    }
    if (player_input.size() != m_playerDim ||
        monster_input.size() != m_monsterEntityNum * m_monsterDim ||
        bullet_input.size() != m_bulletEntityNum * m_bulletDim) {
        UtilityFunctions::push_error(
            "AIAgent.ProcessSensorData: input sizes do not match Init.");
        return output;
    }

    using namespace MNN::Express;
    auto player = _Input({1, 1, m_playerDim}, NCHW);
    auto monster = _Input({1, m_monsterEntityNum, m_monsterDim}, NCHW);
    auto bullet = _Input({1, m_bulletEntityNum, m_bulletDim}, NCHW);
    if (!CopyFlattenedState(player_input, 1, m_playerDim,
                            player->writeMap<float>()) ||
        !CopyFlattenedState(monster_input, m_monsterEntityNum, m_monsterDim,
                            monster->writeMap<float>()) ||
        !CopyFlattenedState(bullet_input, m_bulletEntityNum, m_bulletDim,
                            bullet->writeMap<float>())) {
        UtilityFunctions::push_error(
            "AIAgent.ProcessSensorData: input layout conversion failed.");
        return output;
    }
    const int entity_num = 1 + m_monsterEntityNum + m_bulletEntityNum;
    auto mask = _Input({1, entity_num}, NCHW);
    MiniMind::BuildEntityMask(
        player->readMap<float>(), 1, 1, m_playerDim,
        monster->readMap<float>(), m_monsterEntityNum, m_monsterDim,
        bullet->readMap<float>(), m_bulletEntityNum, m_bulletDim,
        mask->writeMap<float>());
    const auto result = m_mnnActorNet->onForward(
        {player, monster, bullet, mask});
    if (result.size() != 3 || result[0] == nullptr || result[1] == nullptr) {
        UtilityFunctions::push_error("AIAgent.ProcessSensorData: Actor forward failed.");
        return output;
    }
    const float *move = result[0]->readMap<float>();
    const float *shoot = result[1]->readMap<float>();
    if (move == nullptr || shoot == nullptr) {
        UtilityFunctions::push_error("AIAgent.ProcessSensorData: Actor output read failed.");
        return output;
    }

    // 单条推理不加噪声，均值经过 sigmoid 后按三个区间离散化。
    const float horizon = MiniMind::MovePolicy::DeterministicAction(move[0]);
    const float vertical = MiniMind::MovePolicy::DeterministicAction(move[1]);

    const float *monster_mask = mask->readMap<float>() + 1;
    const int target_index = MiniMind::ShootPolicy::MaskedArgmax(
        shoot, monster_mask, m_monsterEntityNum);
    const float shoot_probability =
        1.0f / (1.0f + std::exp(-shoot[m_monsterEntityNum]));

    output.resize(4);
    output[0] = horizon;
    output[1] = vertical;
    output[2] = static_cast<float>(target_index);
    output[3] = shoot_probability > 0.5f ? 1.0f : 0.0f;

    if (isGameEnd) {
        m_mnnActorNet->ResetAllGRUMemory();
    }
    return output;
}

godot::Array godot::AIAgent::BatchProcessSensorData(
    const godot::Array &batch_player,
    const godot::Array &batch_monster,
    const godot::Array &batch_bullet,
    const godot::PackedInt32Array &agent_ids)
{
    godot::Array result_array;
    if (m_mnnActorNet == nullptr || m_mnnCriticNet == nullptr ||
        m_training_data == nullptr) {
        UtilityFunctions::push_error("Agent: training network is not set!");
        return result_array;
    }
    if (m_training_data->buffer_player == nullptr ||
        m_training_data->buffer_monster == nullptr ||
        m_training_data->buffer_bullet == nullptr ||
        m_training_data->buffer_mask == nullptr ||
        m_training_data->buffer_action == nullptr ||
        m_training_data->buffer_log_probs == nullptr ||
        m_training_data->buffer_q_values == nullptr) {
        UtilityFunctions::push_error(
            "AIAgent.BatchProcessSensorData: call SetBatchInfo first.");
        return result_array;
    }
    const int batch_size = batch_player.size();
    if (batch_size == 0 || batch_monster.size() != batch_size ||
        batch_bullet.size() != batch_size || agent_ids.size() != batch_size ||
        batch_size > m_training_data->batch_size) {
        UtilityFunctions::push_error("AIAgent.BatchProcessSensorData: invalid batch sizes.");
        return result_array;
    }

    std::unordered_set<int> unique_agent_ids;
    std::vector<PackedFloat32Array> players, monsters, bullets;
    players.reserve(batch_size); monsters.reserve(batch_size); bullets.reserve(batch_size);
    for (int batch = 0; batch < batch_size; ++batch) {
        if (batch_player[batch].get_type() != Variant::PACKED_FLOAT32_ARRAY ||
            batch_monster[batch].get_type() != Variant::PACKED_FLOAT32_ARRAY ||
            batch_bullet[batch].get_type() != Variant::PACKED_FLOAT32_ARRAY) {
            UtilityFunctions::push_error("AIAgent.BatchProcessSensorData: every input must be PackedFloat32Array.");
            return godot::Array();
        }
        PackedFloat32Array player = batch_player[batch];
        PackedFloat32Array monster = batch_monster[batch];
        PackedFloat32Array bullet = batch_bullet[batch];
        if (player.size() != m_playerDim ||
            monster.size() != m_monsterEntityNum * m_monsterDim ||
            bullet.size() != m_bulletEntityNum * m_bulletDim ||
            !unique_agent_ids.insert(agent_ids[batch]).second) {
            UtilityFunctions::push_error("AIAgent.BatchProcessSensorData: invalid input size or duplicate agent_id.");
            return godot::Array();
        }
        players.push_back(player); monsters.push_back(monster); bullets.push_back(bullet);
    }

    TrainingData::SetZero(m_training_data->buffer_player);
    TrainingData::SetZero(m_training_data->buffer_monster);
    TrainingData::SetZero(m_training_data->buffer_bullet);
    TrainingData::SetZero(m_training_data->buffer_mask);
    TrainingData::SetZero(m_training_data->buffer_action);
    m_training_data->input_mapping.clear();
    float *player_data = m_training_data->buffer_player->writeMap<float>();
    float *monster_data = m_training_data->buffer_monster->writeMap<float>();
    float *bullet_data = m_training_data->buffer_bullet->writeMap<float>();
    for (int batch = 0; batch < batch_size; ++batch) {
        if (!CopyFlattenedState(players[batch], 1, m_playerDim,
                                player_data + batch * m_playerDim) ||
            !CopyFlattenedState(monsters[batch], m_monsterEntityNum, m_monsterDim,
                                monster_data + batch * m_monsterEntityNum * m_monsterDim) ||
            !CopyFlattenedState(bullets[batch], m_bulletEntityNum, m_bulletDim,
                                bullet_data + batch * m_bulletEntityNum * m_bulletDim)) {
            return godot::Array();
        }
        m_training_data->input_mapping[agent_ids[batch]] = batch;
    }
    const int entity_num = 1 + m_monsterEntityNum + m_bulletEntityNum;
    float *mask_data = m_training_data->buffer_mask->writeMap<float>();
    MiniMind::BuildEntityMask(
        player_data, batch_size, 1, m_playerDim,
        monster_data, m_monsterEntityNum, m_monsterDim,
        bullet_data, m_bulletEntityNum, m_bulletDim, mask_data);

    using namespace MNN::Express;
    auto player = _Const(player_data, {batch_size, 1, m_playerDim}, NCHW);
    auto monster = _Const(monster_data,
                          {batch_size, m_monsterEntityNum, m_monsterDim}, NCHW);
    auto bullet = _Const(bullet_data,
                         {batch_size, m_bulletEntityNum, m_bulletDim}, NCHW);
    auto mask = _Const(mask_data, {batch_size, entity_num}, NCHW);
    const auto outputs = m_mnnActorNet->onForward({player, monster, bullet, mask});
    if (outputs.size() != 3 || outputs[0] == nullptr ||
        outputs[1] == nullptr || outputs[2] == nullptr) {
        UtilityFunctions::push_error(Utf8("AIAgent.BatchProcessSensorData：Actor前向计算失败。"));
        return godot::Array();
    }
    const float *move = outputs[0]->readMap<float>();
    const float *shoot = outputs[1]->readMap<float>();
    const auto critic_outputs = m_mnnCriticNet->onForward({outputs[2]});
    if (critic_outputs.size() != 1 || critic_outputs[0] == nullptr ||
        move == nullptr || shoot == nullptr) {
        UtilityFunctions::push_error(Utf8("AIAgent.BatchProcessSensorData：网络输出无效。"));
        return godot::Array();
    }
    const auto critic_output = critic_outputs[0];
    const float *critic = critic_output->readMap<float>();
    if (critic == nullptr) {
        UtilityFunctions::push_error(Utf8("AIAgent.BatchProcessSensorData：Critic输出读取失败。"));
        return godot::Array();
    }
    std::copy(critic, critic + batch_size,
              m_training_data->buffer_q_values->writeMap<float>());

    for (int batch = 0; batch < batch_size; ++batch) {
        const int id = agent_ids[batch];
        const auto write_position = m_training_data->agent_write_index.find(id);
        if (write_position == m_training_data->agent_write_index.end()) {
            m_training_data->agent_write_index[id] = batch;
        } else {
            const int previous_index = write_position->second - m_training_data->batch_size;
            if (previous_index >= 0) {
                m_training_data->old_critic_values->writeMap<float>()[previous_index] = critic[batch];
            }
        }
    }

    static thread_local std::mt19937 generator(std::random_device{}());
    float *buffer_action = m_training_data->buffer_action->writeMap<float>();
    float *buffer_log_probability = m_training_data->buffer_log_probs->writeMap<float>();
    const int move_dim = outputs[0]->getInfo()->dim[1];
    const int shoot_dim = outputs[1]->getInfo()->dim[1];
    result_array.resize(batch_size);

    for (int batch = 0; batch < batch_size; ++batch) {
        const float *move_row = move + batch * move_dim;
        const float *shoot_row = shoot + batch * shoot_dim;
        const float *monster_mask_row = mask_data + batch * entity_num + 1;
        std::uniform_real_distribution<float> uniform(0.0f, 1.0f);
        const float horizontal = MiniMind::MovePolicy::SampleAction(
            move_row[0], m_move_logistic_scale, uniform(generator));
        const float vertical = MiniMind::MovePolicy::SampleAction(
            move_row[1], m_move_logistic_scale, uniform(generator));
        const int target_index = MiniMind::ShootPolicy::SampleCategorical(
            shoot_row, monster_mask_row, m_monsterEntityNum, uniform(generator));
        const float shoot_probability =
            1.0f / (1.0f + std::exp(-shoot_row[m_monsterEntityNum]));
        std::bernoulli_distribution shoot_bernoulli(shoot_probability);
        const float shoot_action = shoot_bernoulli(generator) ? 1.0f : 0.0f;

        // 缓存实际执行的离散移动、怪物索引和连续开火概率。
        float sampled_action[4] = {
            horizontal, vertical, static_cast<float>(target_index), shoot_probability};
        std::copy_n(sampled_action, 4,
                    buffer_action + batch * m_training_data->action_dim);
        const auto sampled_action_tensor = _Const(sampled_action, {1, 4}, NCHW);
        const auto sampled_move_tensor = _Const(move_row, {1, move_dim}, NCHW);
        const auto sampled_shoot_tensor = _Const(shoot_row, {1, shoot_dim}, NCHW);
        const auto sampled_monster_mask = _Const(
            monster_mask_row, {1, m_monsterEntityNum}, NCHW);
        buffer_log_probability[batch] = CalculateLogProbs(
            sampled_action_tensor, sampled_move_tensor, sampled_shoot_tensor,
            sampled_monster_mask, m_move_logistic_scale)->readMap<float>()[0];

        PackedFloat32Array action;
        action.resize(4);
        action[0] = horizontal;
        action[1] = vertical;
        action[2] = static_cast<float>(target_index);
        action[3] = shoot_action;
        result_array[batch] = action;
    }
    if (m_training_data->rollout_memory == nullptr)
    {
        m_training_data->rollout_memory = m_mnnActorNet->GetGRUMemory();
    }
    
    return result_array;
}

void AIAgent::PushTrainingData(
    const godot::PackedFloat32Array& batch_rewards,
    const godot::PackedInt32Array &agent_ids,
    const godot::PackedFloat32Array &batch_dones)
{
    if (m_mnnActorNet == nullptr || m_mnnCriticNet == nullptr ||
        m_training_data == nullptr) {
        UtilityFunctions::push_error("Agent: training network is not set!");
        return;
    }
    if (m_training_data->player_state == nullptr ||
        m_training_data->monster_state == nullptr ||
        m_training_data->bullet_state == nullptr ||
        m_training_data->mask == nullptr ||
        m_training_data->actions == nullptr ||
        m_training_data->rewards == nullptr ||
        m_training_data->done == nullptr ||
        m_training_data->old_log_probs == nullptr ||
        m_training_data->old_critic_values == nullptr ||
        m_training_data->old_q_values == nullptr ||
        m_training_data->buffer_player == nullptr ||
        m_training_data->buffer_monster == nullptr ||
        m_training_data->buffer_bullet == nullptr ||
        m_training_data->buffer_mask == nullptr ||
        m_training_data->buffer_action == nullptr ||
        m_training_data->buffer_log_probs == nullptr ||
        m_training_data->buffer_q_values == nullptr) {
        UtilityFunctions::push_error(Utf8("AIAgent.PushTrainingData：请先调用SetBatchInfo。"));
        return;
    }
    if (batch_rewards.size() == 0 || agent_ids.size() != batch_rewards.size() ||
        batch_dones.size() != batch_rewards.size()) {
        UtilityFunctions::push_error("Agent: invalid training data batch!");
        return;
    }
    if (batch_rewards.size() > m_training_data->batch_size) {
        UtilityFunctions::push_error(
            Utf8("AIAgent.PushTrainingData：当前 batch 超过 SetBatchInfo 配置的容量。"));
        return;
    }

    // 在写入训练帧前一次性校验映射和下标，防止部分样本写入后留下不完整帧。
    std::unordered_set<int> unique_agent_ids;
    for (int batch = 0; batch < batch_rewards.size(); ++batch) {
        const int id = agent_ids[batch];
        if (!unique_agent_ids.insert(id).second) {
            UtilityFunctions::push_error(
                Utf8("AIAgent.PushTrainingData：同一 batch 中的 agent_id 不能重复。"));
            return;
        }
        const auto write_it = m_training_data->agent_write_index.find(id);
        const auto input_it = m_training_data->input_mapping.find(id);
        if (write_it == m_training_data->agent_write_index.end() ||
            input_it == m_training_data->input_mapping.end()) {
            UtilityFunctions::push_error(
                Utf8("AIAgent.PushTrainingData：训练数据与上一批前向结果的 agent_id 不匹配。"));
            return;
        }
        const int sample_capacity =
            m_training_data->batch_size * m_training_data->num_frames;
        if (write_it->second < 0 || write_it->second >= sample_capacity ||
            input_it->second < 0 ||
            input_it->second >= m_training_data->batch_size) {
            UtilityFunctions::push_error(
                Utf8("AIAgent.PushTrainingData：训练帧下标超出已配置容量。"));
            return;
        }
    }

    const int player_size = m_playerDim;
    const int monster_size = m_monsterEntityNum * m_monsterDim;
    const int bullet_size = m_bulletEntityNum * m_bulletDim;
    const int entity_num = 1 + m_monsterEntityNum + m_bulletEntityNum;
    float *player_states = m_training_data->player_state->writeMap<float>();
    float *monster_states = m_training_data->monster_state->writeMap<float>();
    float *bullet_states = m_training_data->bullet_state->writeMap<float>();
    float *masks = m_training_data->mask->writeMap<float>();
    float *actions = m_training_data->actions->writeMap<float>();
    float *old_log_probs = m_training_data->old_log_probs->writeMap<float>();
    float *rewards = m_training_data->rewards->writeMap<float>();
    float *dones = m_training_data->done->writeMap<float>();
    float *old_q_values = m_training_data->old_q_values->writeMap<float>();
    const float *buffer_players = m_training_data->buffer_player->readMap<float>();
    const float *buffer_monsters = m_training_data->buffer_monster->readMap<float>();
    const float *buffer_bullets = m_training_data->buffer_bullet->readMap<float>();
    const float *buffer_masks = m_training_data->buffer_mask->readMap<float>();
    const float *buffer_actions = m_training_data->buffer_action->readMap<float>();
    const float *buffer_log_probs = m_training_data->buffer_log_probs->readMap<float>();
    const float *buffer_q_values = m_training_data->buffer_q_values->readMap<float>();

    for (int batch = 0; batch < batch_rewards.size(); ++batch) {
        const int id = agent_ids[batch];
        const auto write_it = m_training_data->agent_write_index.find(id);
        const auto input_it = m_training_data->input_mapping.find(id);
        const int index = write_it->second;
        const int buffer_index = input_it->second;
        write_it->second += m_training_data->batch_size;

        std::copy_n(buffer_players + buffer_index * player_size, player_size,
                    player_states + index * player_size);
        std::copy_n(buffer_monsters + buffer_index * monster_size, monster_size,
                    monster_states + index * monster_size);
        std::copy_n(buffer_bullets + buffer_index * bullet_size, bullet_size,
                    bullet_states + index * bullet_size);
        std::copy_n(buffer_masks + buffer_index * entity_num, entity_num,
                    masks + index * entity_num);
        std::copy_n(buffer_actions + buffer_index * m_training_data->action_dim,
                    m_training_data->action_dim,
                    actions + index * m_training_data->action_dim);
        old_log_probs[index] = buffer_log_probs[buffer_index];
        rewards[index] = batch_rewards[batch];
        dones[index] = batch_dones[batch];
        old_q_values[index] = buffer_q_values[buffer_index];

        if (batch_dones[batch] > 0.0f) {
            m_mnnActorNet->ResetGRUMemory(buffer_index);
        }
    }
}

void AIAgent::Train(int step)
{
    if (mode != AIAgentMode::TRAINING || m_training_data == nullptr ||
        m_mnnActorNet == nullptr || m_mnnCriticNet == nullptr ||
        m_actor_optimizer == nullptr || m_critic_optimizer == nullptr) {
        UtilityFunctions::push_error(Utf8("AIAgent.Train：训练网络尚未初始化。"));
        return;
    }
    if (step <= 0) {
        UtilityFunctions::push_error(Utf8("AIAgent.Train：step必须为正整数。"));
        return;
    }
    if (m_training_data->player_state == nullptr ||
        m_training_data->monster_state == nullptr ||
        m_training_data->bullet_state == nullptr ||
        m_training_data->mask == nullptr ||
        m_training_data->actions == nullptr ||
        m_training_data->rewards == nullptr ||
        m_training_data->done == nullptr ||
        m_training_data->old_log_probs == nullptr ||
        m_training_data->old_critic_values == nullptr ||
        m_training_data->old_q_values == nullptr) {
        UtilityFunctions::push_error(Utf8("AIAgent.Train：请先调用SetBatchInfo并写入训练数据。"));
        return;
    }
    using namespace MNN::Express;
    const int batch_size = m_training_data->batch_size;
    const int frame_count = m_training_data->num_frames;
    const int sample_count = batch_size * frame_count;
    // 一次 Train 的全部 PPO 更新共用采集这批数据时的探索范围。
    const float rollout_move_scale = m_move_logistic_scale;

    const auto td_target = m_training_data->rewards
        + _Scalar<float>(m_gamma) * m_training_data->old_critic_values
        * (_Scalar<float>(1.0f) - m_training_data->done);
    const auto td_delta = td_target - m_training_data->old_q_values;
    auto advantage = _Input({sample_count, 1}, NCHW);
    float *advantage_values = advantage->writeMap<float>();

    // 每个智能体沿时间轴单独计算 GAE，不能把 batch 维当作时间维。
    const float *delta_values = td_delta->readMap<float>();
    const float *done_values = m_training_data->done->readMap<float>();
    for (int agent = 0; agent < batch_size; ++agent) {
        std::vector<float> agent_delta_values(static_cast<std::size_t>(frame_count));
        std::vector<float> agent_done_values(static_cast<std::size_t>(frame_count));
        for (int frame = 0; frame < frame_count; ++frame) {
            const int index = frame * batch_size + agent;
            agent_delta_values[frame] = delta_values[index];
            agent_done_values[frame] = done_values[index];
        }
        const auto agent_delta = _Const(
            agent_delta_values.data(), {frame_count, 1}, NCHW);
        const auto agent_done = _Const(
            agent_done_values.data(), {frame_count, 1}, NCHW);
        const auto agent_advantage = ComputeAdvantage(
            agent_delta, agent_done, m_gamma, m_lambda);
        const float *agent_advantage_values = agent_advantage->readMap<float>();
        for (int frame = 0; frame < frame_count; ++frame) {
            advantage_values[frame * batch_size + agent] =
                agent_advantage_values[frame];
        }
    }
    m_mnnActorNet->CacheMemory();
    for (int epoch = 0; epoch < step; ++epoch) {
        // m_mnnActorNet->ResetAllGRUMemory();
        m_mnnActorNet->SetGRUMemory(m_training_data->rollout_memory);
        std::vector<VARP> frame_log_probabilities;
        std::vector<VARP> frame_q_values;
        frame_log_probabilities.reserve(frame_count);
        frame_q_values.reserve(frame_count);

        for (int frame = 0; frame < frame_count; ++frame) {
            const int offset = frame * batch_size;
            auto players = Slice2D(
                _Reshape(m_training_data->player_state, {sample_count, m_playerDim}),
                offset, 0, batch_size, m_playerDim);
            players = _Reshape(players, {batch_size, 1, m_playerDim});
            auto monsters = Slice2D(
                _Reshape(m_training_data->monster_state,
                         {sample_count, m_monsterEntityNum * m_monsterDim}),
                offset, 0, batch_size, m_monsterEntityNum * m_monsterDim);
            monsters = _Reshape(monsters,
                {batch_size, m_monsterEntityNum, m_monsterDim});
            auto bullets = Slice2D(
                _Reshape(m_training_data->bullet_state,
                         {sample_count, m_bulletEntityNum * m_bulletDim}),
                offset, 0, batch_size, m_bulletEntityNum * m_bulletDim);
            bullets = _Reshape(bullets,
                {batch_size, m_bulletEntityNum, m_bulletDim});
            const int entity_num = 1 + m_monsterEntityNum + m_bulletEntityNum;
            auto masks = Slice2D(
                m_training_data->mask, offset, 0, batch_size, entity_num);
            auto actions = Slice2D(
                m_training_data->actions, offset, 0, batch_size,
                m_training_data->action_dim);
            auto dones = Slice2D(
                m_training_data->done, offset, 0, batch_size, 1);

            const auto actor_outputs = m_mnnActorNet->onForward({players, monsters, bullets, masks});
            auto monster_masks = Slice2D(
                masks, 0, 1, batch_size, m_monsterEntityNum);
            frame_log_probabilities.push_back(CalculateLogProbs(
                actions, actor_outputs[0], actor_outputs[1], monster_masks,
                rollout_move_scale));
            frame_q_values.push_back(m_mnnCriticNet->onForward({actor_outputs[2]})[0]);

            const float *done_values = dones->readMap<float>();
            for (int agent = 0; agent < batch_size; ++agent) {
                if (done_values[agent] > 0.0f) {
                    m_mnnActorNet->ResetGRUMemory(agent);
                }
            }
        }

        const auto batch_log_probabilities = _Concat(frame_log_probabilities, 0);
        const auto batch_new_q_values = _Concat(frame_q_values, 0);
        const auto ratio = _Exp(batch_log_probabilities - m_training_data->old_log_probs);
        const auto clipped_ratio = _Minimum(
            _Maximum(ratio, _Scalar<float>(1.0f - m_clip_epsilon)),
            _Scalar<float>(1.0f + m_clip_epsilon));
        const auto surrogate = _Minimum(ratio * advantage, clipped_ratio * advantage);
        const auto actor_loss = _Negative(_ReduceMean(surrogate, {}));
        const auto critic_difference = batch_new_q_values - td_target;
        const auto critic_loss = _ReduceMean(_Square(critic_difference), {});

        m_actor_optimizer->step(actor_loss);
        m_critic_optimizer->step(critic_loss);        
    }
    m_training_data->ClearTrainingData();
    m_mnnActorNet->RestoreMemory();
    // 整次 Train 完成后只衰减一次，step 不影响衰减次数。
    m_move_logistic_scale = MiniMind::MovePolicy::DecayScale(
        m_move_logistic_scale);
}

void godot::AIAgent::SetBatchInfo(int batch_size, int action_dim, int num_frames)
{
    if (batch_size <= 0 || num_frames <= 0 || action_dim != 4) {
        UtilityFunctions::push_error(
            Utf8("AIAgent.SetBatchInfo：batch_size和num_frames必须为正数，action_dim必须为4。"));
        return;
    }
    if (mode == AIAgentMode::TRAINING && m_training_data != nullptr) {
        m_training_data->Init(
            batch_size, num_frames, m_monsterEntityNum, m_bulletEntityNum,
            m_playerDim, m_monsterDim, m_bulletDim, action_dim);
    } else if (mode == AIAgentMode::TRAINING) {
        UtilityFunctions::push_error(Utf8("AIAgent.SetBatchInfo：请先调用Init初始化训练网络。"));
        return;
    }
    if (m_mnnActorNet != nullptr) {
        m_mnnActorNet->ResetAllGRUMemory();
    }
}

void godot::AIAgent::SetLearningParameters(
    float gamma, float lambda, float clip_epsilon, float continuous_gamma)
{
    if (gamma < 0.0f || lambda < 0.0f || lambda > 1.0f ||
        clip_epsilon < 0.0f || continuous_gamma < 0.0f) {
        UtilityFunctions::push_error(Utf8("AIAgent.SetLearningParameters：学习参数超出有效范围。"));
        return;
    }
    m_gamma = gamma;
    m_lambda = lambda;
    m_clip_epsilon = clip_epsilon;
    m_continuous_gamma = continuous_gamma;
}

namespace {

String MakeActorModelName(const String &file_name)
{
    // Actor 严格使用调用方给出的文件名，不在内部追加或替换扩展名。
    return file_name;
}

String MakeCriticModelName(const String &actor_file_name)
{
    const String extension = actor_file_name.get_extension();
    if (extension.is_empty()) {
        return actor_file_name + String("_critic");
    }
    return actor_file_name.get_basename() + String("_critic.") + extension;
}

String MakeModelPath(const String &parent_folder, const String &file_name)
{
    return parent_folder.is_empty()
        ? file_name
        : parent_folder.path_join(file_name);
}

bool EnsureModelDirectory(const String &model_path)
{
    const String directory = model_path.get_base_dir();
    if (directory.is_empty() || directory == ".") {
        return true;
    }
    const String absolute_directory =
        ProjectSettings::get_singleton()->globalize_path(directory);
    if (DirAccess::dir_exists_absolute(absolute_directory)) {
        return true;
    }
    return DirAccess::make_dir_recursive_absolute(absolute_directory) == OK;
}

bool SaveModuleParameters(const String &model_path, MNN::Express::Module *module)
{
    if (module == nullptr || !EnsureModelDirectory(model_path)) {
        return false;
    }

    // 参数是彼此独立的图输出，显式命名可避免 MNN 按执行拓扑保存后改变恢复顺序。
    const std::vector<MNN::Express::VARP> parameters = module->parameters();
    for (std::size_t index = 0; index < parameters.size(); ++index) {
        parameters[index]->setName(
            "minimind_parameter_" + std::to_string(index));
    }

    // 使用 MNN 原生 FlatBuffer 模型格式序列化模块参数，Godot 只负责路径和文件写入。
    const std::vector<int8_t> serialized =
        MNN::Express::Variable::save(parameters);
    if (serialized.empty()) {
        return false;
    }

    Ref<FileAccess> file = FileAccess::open(model_path, FileAccess::WRITE);
    if (file.is_null()) {
        return false;
    }
    PackedByteArray bytes;
    bytes.resize(static_cast<int64_t>(serialized.size()));
    std::copy(serialized.begin(), serialized.end(), bytes.ptrw());
    const bool stored = file->store_buffer(bytes);
    file->flush();
    const Error write_error = file->get_error();
    file->close();
    return stored && write_error == OK &&
        FileAccess::get_size(model_path) == serialized.size();
}

bool LoadModuleParameters(const String &model_path, MNN::Express::Module *module)
{
    if (module == nullptr) {
        return false;
    }
    Ref<FileAccess> file = FileAccess::open(model_path, FileAccess::READ);
    if (file.is_null()) {
        return false;
    }
    const uint64_t byte_count = file->get_length();
    if (byte_count == 0 || byte_count > static_cast<uint64_t>(INT64_MAX)) {
        file->close();
        return false;
    }
    const PackedByteArray bytes = file->get_buffer(
        static_cast<int64_t>(byte_count));
    file->close();
    if (static_cast<uint64_t>(bytes.size()) != byte_count) {
        return false;
    }

    // MNN 负责解析模型；再按保存时的显式名称还原模块参数注册顺序。
    const std::map<std::string, MNN::Express::VARP> parameter_map =
        MNN::Express::Variable::loadMap(
            bytes.ptr(), static_cast<std::size_t>(bytes.size()));
    const std::size_t expected_count = module->parameters().size();
    std::vector<MNN::Express::VARP> parameters;
    parameters.reserve(expected_count);
    for (std::size_t index = 0; index < expected_count; ++index) {
        const std::string name =
            "minimind_parameter_" + std::to_string(index);
        const auto found = parameter_map.find(name);
        if (found == parameter_map.end() || found->second == nullptr) {
            return false;
        }
        parameters.push_back(found->second);
    }
    // loadParameters 继续由 MNN 原生接口校验维度、布局和数据类型。
    return module->loadParameters(parameters);
}

} // 匿名命名空间

void godot::AIAgent::Save(
    const godot::String &parent_folder, const godot::String &file_name)
{
    if (m_mnnActorNet == nullptr) {
        UtilityFunctions::push_error(Utf8("AIAgent.Save：网络尚未初始化。"));
        return;
    }
    if (mode == AIAgentMode::TRAINING && m_mnnCriticNet == nullptr) {
        UtilityFunctions::push_error(Utf8("AIAgent.Save：训练模式下 Critic 尚未初始化。"));
        return;
    }

    const String actor_name = MakeActorModelName(file_name);
    if (actor_name.is_empty()) {
        UtilityFunctions::push_error(Utf8("AIAgent.Save：模型文件名不能为空。"));
        return;
    }
    const String actor_path = MakeModelPath(parent_folder, actor_name);
    if (!SaveModuleParameters(actor_path, m_mnnActorNet)) {
        UtilityFunctions::push_error(
            Utf8("AIAgent.Save：Actor MNN 模型保存失败：") + actor_path);
        return;
    }

    if (mode == AIAgentMode::TRAINING) {
        const String critic_path = MakeModelPath(
            parent_folder, MakeCriticModelName(actor_name));
        if (!SaveModuleParameters(critic_path, m_mnnCriticNet)) {
            UtilityFunctions::push_error(
                Utf8("AIAgent.Save：Critic MNN 模型保存失败：") + critic_path);
        }
    }
}

void godot::AIAgent::Load(
    const godot::String &parent_folder, const godot::String &file_name)
{
    if (m_mnnActorNet == nullptr) {
        UtilityFunctions::push_error(Utf8("AIAgent.Load：网络尚未初始化。"));
        return;
    }
    if (mode == AIAgentMode::TRAINING && m_mnnCriticNet == nullptr) {
        UtilityFunctions::push_error(Utf8("AIAgent.Load：训练模式下 Critic 尚未初始化。"));
        return;
    }

    const String actor_name = MakeActorModelName(file_name);
    if (actor_name.is_empty()) {
        UtilityFunctions::push_error(Utf8("AIAgent.Load：模型文件名不能为空。"));
        return;
    }
    const String actor_path = MakeModelPath(parent_folder, actor_name);
    if (!LoadModuleParameters(actor_path, m_mnnActorNet)) {
        UtilityFunctions::push_error(
            Utf8("AIAgent.Load：Actor MNN 模型读取或参数匹配失败：") + actor_path);
        return;
    }

    if (mode == AIAgentMode::TRAINING) {
        const String critic_path = MakeModelPath(
            parent_folder, MakeCriticModelName(actor_name));
        if (!LoadModuleParameters(critic_path, m_mnnCriticNet)) {
            UtilityFunctions::push_error(
                Utf8("AIAgent.Load：Critic MNN 模型读取或参数匹配失败：") + critic_path);
            return;
        }
    }
    m_mnnActorNet->ResetAllGRUMemory();
}
