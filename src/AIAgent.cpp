#include "AIAgent.h"
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
    ClassDB::bind_method(D_METHOD("Init", "entity_num", "feature_dim", "move_dim", "shoot_dim", "embedding_dim", "attention_key_dim", "gru_hidden_dim", "out_hidden_dim"), &AIAgent::Init, DEFVAL(16), DEFVAL(16), DEFVAL(128), DEFVAL(128));
    ClassDB::bind_method(D_METHOD("get_mode"), &AIAgent::get_mode);
    ClassDB::bind_method(D_METHOD("set_mode", "mode"), &AIAgent::set_mode);
    ClassDB::bind_method(D_METHOD("ProcessSensorData", "data", "isGameEnd"), &AIAgent::ProcessSensorData, DEFVAL(false));
    ClassDB::bind_method(D_METHOD("BatchProcessSensorData", "batch_data", "agent_ids"), &AIAgent::BatchProcessSensorData);
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
    const MNN::Express::VARP &shoot_data)
{
    using namespace MNN::Express;
    if (actions == nullptr || move_data == nullptr || shoot_data == nullptr) {
        UtilityFunctions::push_error(Utf8("CalculateLogProbs：输入不能为空。"));
        return nullptr;
    }
    const auto *action_info = actions->getInfo();
    const auto *move_info = move_data->getInfo();
    const auto *shoot_info = shoot_data->getInfo();
    if (action_info == nullptr || move_info == nullptr || shoot_info == nullptr ||
        action_info->dim.size() != 2 || move_info->dim.size() != 2 ||
        shoot_info->dim.size() != 2 || action_info->dim[0] != move_info->dim[0] ||
        action_info->dim[0] != shoot_info->dim[0] || action_info->dim[1] < 4 ||
        move_info->dim[1] < 6 || shoot_info->dim[1] < 3) {
        UtilityFunctions::push_error(Utf8("CalculateLogProbs：输入形状不匹配。"));
        return nullptr;
    }
    const int batch_size = action_info->dim[0];

    auto categorical_log_probability = [&](int move_offset, int action_column) {
        const auto logits = Slice2D(move_data, 0, move_offset, batch_size, 3);
        const auto maximum = _ReduceMax(logits, {1}, true);
        const auto log_normalizer = maximum + _Log(
            _ReduceSum(_Exp(logits - maximum), {1}, true));
        const auto action = _Reshape(
            _Cast(Slice2D(actions, 0, action_column, batch_size, 1),
                  halide_type_of<int>()),
            {batch_size});
        const auto one_hot = _OneHot(
            action, _Scalar<int>(3), _Scalar<float>(1.0f),
            _Scalar<float>(0.0f));
        return _ReduceSum((logits - log_normalizer) * one_hot, {1}, true);
    };

    const auto horizontal_log_probability = categorical_log_probability(0, 0);
    const auto vertical_log_probability = categorical_log_probability(3, 1);

    const auto mean_x = Slice2D(shoot_data, 0, 0, batch_size, 1);
    const auto mean_y = Slice2D(shoot_data, 0, 1, batch_size, 1);
    const auto variance = Slice2D(shoot_data, 0, 2, batch_size, 1);
    const auto standard_deviation = _Sqrt(_Abs(variance) + _Scalar<float>(1.0e-6f));
    const auto mean_angle = _Atan2(mean_y, mean_x);
    const auto action_x = Slice2D(actions, 0, 2, batch_size, 1);
    const auto action_y = Slice2D(actions, 0, 3, batch_size, 1);
    const auto action_angle = _Atan2(action_y, action_x);
    const auto raw_delta = action_angle - mean_angle;
    const auto wrapped_delta = _Atan2(_Sin(raw_delta), _Cos(raw_delta));
    const auto normalized_delta = wrapped_delta / standard_deviation;
    const auto angle_log_probability =
        _Scalar<float>(-0.5f) * _Square(normalized_delta)
        - _Log(standard_deviation)
        - _Scalar<float>(0.9189385332046727f);

    const auto shoot_logit = Slice2D(shoot_data, 0, 3, batch_size, 1);
    const auto log_shoot = log_sigmoid(shoot_logit);

    return horizontal_log_probability + vertical_log_probability
        + angle_log_probability + log_shoot;
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
    m_entityNum = 0;
    m_entityFeatureDim = 0;
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
    int entity_num, int feature_dim, int move_dim, int shoot_dim,
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
    m_entityNum = 0;
    m_entityFeatureDim = 0;

    if (entity_num <= 0 || feature_dim <= 0 || move_dim < 6 ||
        shoot_dim < 4 || embedding_dim <= 0 || attention_key_dim <= 0 ||
        gru_hidden_dim <= 0 || out_hidden_dim <= 0) {
        UtilityFunctions::push_error(
            Utf8("AIAgent.Init：维度必须为正数，move_dim至少为6，shoot_dim至少为4。"));
        return;
    }

    m_insize = entity_num * feature_dim;
    m_outSize = move_dim + shoot_dim;
    m_entityNum = entity_num;
    m_entityFeatureDim = feature_dim;

    m_mnnActorNet = new MiniMind::ActorNet(
        feature_dim, move_dim, shoot_dim, embedding_dim,
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
    const PackedFloat32Array &input, bool isGameEnd)
{
    PackedFloat32Array output;
    if (m_mnnActorNet == nullptr) {
        UtilityFunctions::push_error(Utf8("AIAgent.ProcessSensorData：Actor 尚未初始化。"));
        return output;
    }
    if (input.size() != m_insize) {
        UtilityFunctions::push_error(
            Utf8("AIAgent.ProcessSensorData：输入长度必须等于 entity_num*feature_dim。"));
        return output;
    }

    using namespace MNN::Express;
    // PackedFloat32Array 是一个 [E*F] 拼接向量，在写入后明确赋予 [1,E,F] 形状。
    auto state = _Input({1, m_entityNum, m_entityFeatureDim}, NCHW);
    if (!CopyFlattenedState(input, m_entityNum, m_entityFeatureDim,
                            state->writeMap<float>())) {
        UtilityFunctions::push_error(
            Utf8("AIAgent.ProcessSensorData：输入必须是按实体连续拼接的 [entity*feature] 向量。"));
        return output;
    }
    const auto result = m_mnnActorNet->onForward({state});
    if (result.size() != 3 || result[0] == nullptr || result[1] == nullptr) {
        UtilityFunctions::push_error(Utf8("AIAgent.ProcessSensorData：Actor前向计算失败。"));
        return output;
    }
    const float *move = result[0]->readMap<float>();
    const float *shoot = result[1]->readMap<float>();
    if (move == nullptr || shoot == nullptr) {
        UtilityFunctions::push_error(Utf8("AIAgent.ProcessSensorData：Actor输出读取失败。"));
        return output;
    }

    int horizon = 0;
    int vertical = 0;
    if (result[0]->getInfo()->dim[1] >= 3) {
        horizon = static_cast<int>(std::max_element(move, move + 3) - move);
    }
    if (result[0]->getInfo()->dim[1] >= 6) {
        vertical = static_cast<int>(std::max_element(move + 3, move + 6) - (move + 3));
    }

    float angle_x = shoot[0];
    float angle_y = shoot[1];
    const float length = std::sqrt(angle_x * angle_x + angle_y * angle_y);
    if (length > 0.0f) {
        angle_x /= length;
        angle_y /= length;
    }
    const float shoot_probability = 1.0f / (1.0f + std::exp(-shoot[3]));

    output.resize(5);
    output[0] = static_cast<float>(horizon);
    output[1] = static_cast<float>(vertical);
    output[2] = angle_x;
    output[3] = angle_y;
    output[4] = shoot_probability > 0.5f ? 1.0f : 0.0f;

    if (isGameEnd) {
        m_mnnActorNet->ResetAllGRUMemory();
    }
    return output;
}

godot::Array godot::AIAgent::BatchProcessSensorData(
    const godot::Array &batch_data,
    const godot::PackedInt32Array &agent_ids)
{
    godot::Array result_array;
    if (m_mnnActorNet == nullptr || m_mnnCriticNet == nullptr ||
        m_training_data == nullptr) {
        UtilityFunctions::push_error("Agent: training network is not set!");
        return result_array;
    }
    if (m_training_data->buffer_input == nullptr ||
        m_training_data->buffer_action == nullptr ||
        m_training_data->buffer_log_probs == nullptr ||
        m_training_data->buffer_q_values == nullptr) {
        UtilityFunctions::push_error(
            Utf8("AIAgent.BatchProcessSensorData：请先调用SetBatchInfo。"));
        return result_array;
    }
    if (batch_data.size() == 0 || agent_ids.size() != batch_data.size()) {
        UtilityFunctions::push_error("Agent: invalid batch input!");
        return result_array;
    }

    const int batch_size = batch_data.size();
    if (batch_size > m_training_data->batch_size) {
        UtilityFunctions::push_error("Agent: batch size exceeds configured capacity!");
        return result_array;
    }

    // 先完整校验，避免无效 batch 写入一半后污染本帧缓存和智能体映射。
    std::unordered_set<int> unique_agent_ids;
    std::vector<PackedFloat32Array> samples;
    samples.reserve(static_cast<std::size_t>(batch_size));
    for (int batch = 0; batch < batch_size; ++batch) {
        const Variant sample_variant = batch_data[batch];
        if (sample_variant.get_type() != Variant::PACKED_FLOAT32_ARRAY) {
            UtilityFunctions::push_error(
                Utf8("AIAgent.BatchProcessSensorData：batch 中的每个样本都必须是 PackedFloat32Array。"));
            return godot::Array();
        }
        const PackedFloat32Array sample = sample_variant;
        if (sample.size() != m_insize) {
            UtilityFunctions::push_error(
                Utf8("AIAgent.BatchProcessSensorData：每个样本必须包含 entity_num*feature_dim 个数值。"));
            return godot::Array();
        }
        if (!unique_agent_ids.insert(agent_ids[batch]).second) {
            UtilityFunctions::push_error(
                Utf8("AIAgent.BatchProcessSensorData：同一 batch 中的 agent_id 不能重复。"));
            return godot::Array();
        }
        samples.push_back(sample);
    }

    TrainingData::SetZero(m_training_data->buffer_input);
    TrainingData::SetZero(m_training_data->buffer_action);
    m_training_data->input_mapping.clear();
    float *state_data = m_training_data->buffer_input->writeMap<float>();
    for (int batch = 0; batch < batch_size; ++batch) {
        // batch 偏移在最外层，每个样本内部仍是 entity 优先、feature 连续。
        if (!CopyFlattenedState(samples[batch], m_entityNum,
                                m_entityFeatureDim,
                                state_data + batch * m_insize)) {
            UtilityFunctions::push_error(
                Utf8("AIAgent.BatchProcessSensorData：样本布局转换失败。"));
            return godot::Array();
        }
        m_training_data->input_mapping[agent_ids[batch]] = batch;
    }

    using namespace MNN::Express;
    // 只取当前实际 batch 的连续缓存，并明确构造为 [B,E,F]。
    auto state = _Const(state_data,
                        {batch_size, m_entityNum, m_entityFeatureDim}, NCHW);
    const auto outputs = m_mnnActorNet->onForward({state});
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
        std::discrete_distribution<int> horizontal_distribution({
            std::exp(move_row[0]), std::exp(move_row[1]), std::exp(move_row[2])});
        std::discrete_distribution<int> vertical_distribution({
            std::exp(move_row[3]), std::exp(move_row[4]), std::exp(move_row[5])});
        const float horizontal = static_cast<float>(horizontal_distribution(generator));
        const float vertical = static_cast<float>(vertical_distribution(generator));

        const float standard_deviation = std::sqrt(std::abs(shoot_row[2]) + 1.0e-6f);
        std::normal_distribution<float> normal(0.0f, 1.0f);
        float angle_x = shoot_row[0] + standard_deviation * normal(generator);
        float angle_y = shoot_row[1] + standard_deviation * normal(generator);
        const float length = std::sqrt(angle_x * angle_x + angle_y * angle_y);

        float sampled_action[5] = {
            horizontal, vertical, angle_x, angle_y, shoot_row[3]};
        std::copy_n(sampled_action, m_training_data->action_dim,
                    buffer_action + batch * m_training_data->action_dim);
        const auto sampled_action_tensor = _Const(
            sampled_action, {1, 5}, NCHW);
        const auto sampled_move_tensor = _Const(
            move_row, {1, move_dim}, NCHW);
        const auto sampled_shoot_tensor = _Const(
            shoot_row, {1, shoot_dim}, NCHW);
        buffer_log_probability[batch] = CalculateLogProbs(
            sampled_action_tensor, sampled_move_tensor,
            sampled_shoot_tensor)->readMap<float>()[0];

        if (length > 1.0e-6f) {
            angle_x /= length;
            angle_y /= length;
        } else {
            angle_x = 1.0f;
            angle_y = 0.0f;
        }
        const float shoot_probability = 1.0f / (1.0f + std::exp(-shoot_row[3]));
        std::bernoulli_distribution shoot_bernoulli(shoot_probability);
        PackedFloat32Array action;
        action.resize(5);
        action[0] = horizontal;
        action[1] = vertical;
        action[2] = angle_x;
        action[3] = angle_y;
        action[4] = shoot_bernoulli(generator) ? 1.0f : 0.0f;
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
    if (m_training_data->state == nullptr ||
        m_training_data->actions == nullptr ||
        m_training_data->rewards == nullptr ||
        m_training_data->done == nullptr ||
        m_training_data->old_log_probs == nullptr ||
        m_training_data->old_critic_values == nullptr ||
        m_training_data->old_q_values == nullptr ||
        m_training_data->buffer_input == nullptr ||
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

    const int state_size = m_training_data->entity_num * m_training_data->feature_dim;
    float *states = m_training_data->state->writeMap<float>();
    float *actions = m_training_data->actions->writeMap<float>();
    float *old_log_probs = m_training_data->old_log_probs->writeMap<float>();
    float *rewards = m_training_data->rewards->writeMap<float>();
    float *dones = m_training_data->done->writeMap<float>();
    float *old_q_values = m_training_data->old_q_values->writeMap<float>();
    const float *buffer_states = m_training_data->buffer_input->readMap<float>();
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

        std::copy_n(buffer_states + buffer_index * state_size, state_size,
                    states + index * state_size);
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
    if (m_training_data->state == nullptr ||
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
            auto states = Slice2D(
                _Reshape(m_training_data->state,
                         {sample_count, m_entityNum * m_entityFeatureDim}),
                offset, 0, batch_size, m_entityNum * m_entityFeatureDim);
            states = _Reshape(states,
                              {batch_size, m_entityNum, m_entityFeatureDim});
            auto actions = Slice2D(
                m_training_data->actions, offset, 0, batch_size,
                m_training_data->action_dim);
            auto dones = Slice2D(
                m_training_data->done, offset, 0, batch_size, 1);

            const auto actor_outputs = m_mnnActorNet->onForward({states});
            frame_log_probabilities.push_back(CalculateLogProbs(
                actions, actor_outputs[0], actor_outputs[1]));
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
}

void godot::AIAgent::SetBatchInfo(int batch_size, int action_dim, int num_frames)
{
    if (batch_size <= 0 || num_frames <= 0 || action_dim != 5) {
        UtilityFunctions::push_error(
            Utf8("AIAgent.SetBatchInfo：batch_size和num_frames必须为正数，action_dim必须为5。"));
        return;
    }
    if (mode == AIAgentMode::TRAINING && m_training_data != nullptr) {
        m_training_data->Init(
            batch_size, num_frames, m_entityNum, m_entityFeatureDim, action_dim);
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
