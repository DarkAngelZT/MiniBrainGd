#include "ActorNet.h"

#include "attention.hpp"
#include "embedding.hpp"
#include "gru.hpp"
#include "state_pooling.hpp"

#include <MNN/expr/ExprCreator.hpp>
#include <MNN/MNNDefine.h>

#include <algorithm>
#include <random>
#include <utility>
#include <vector>

namespace MiniMind {
namespace {

MNN::Express::VARP make_parameter(const std::vector<int>& shape) {
    std::size_t element_count = 1;
    for (const int extent : shape) {
        element_count *= static_cast<std::size_t>(extent);
    }

    static thread_local std::mt19937 generator(std::random_device{}());
    std::normal_distribution<float> distribution(0.0f, 0.1f);
    std::vector<float> values(element_count);
    for (float& value : values) {
        value = distribution(generator);
    }
    return MNN::Express::_TrainableParam(
        values.data(), shape, MNN::Express::NCHW);
}

void initialize_normal(const std::shared_ptr<MNN::Express::Module>& module) {
    static thread_local std::mt19937 generator(std::random_device{}());
    std::normal_distribution<float> distribution(0.0f, 0.1f);
    for (const auto& parameter : module->parameters()) {
        const auto* info = parameter->getInfo();
        float* values = parameter->writeMap<float>();
        for (std::size_t index = 0; index < info->size; ++index) {
            values[index] = distribution(generator);
        }
    }
}

template<typename ModuleType>
std::shared_ptr<ModuleType> clone_child(
    const std::shared_ptr<ModuleType>& source,
    MNN::Express::Module::CloneContext* context) {
    auto* cloned = dynamic_cast<ModuleType*>(source->clone(context));
    if (cloned == nullptr) {
        MNN_ERROR("ActorNet：子模块克隆失败。\n");
        return nullptr;
    }
    return std::shared_ptr<ModuleType>(cloned);
}

} // 匿名命名空间

ActorNet::ActorNet(int player_dim,
                   int monster_dim,
                   int bullet_dim,
                   int monster_entity_num,
                   int move_output_size)
    : ActorNet(player_dim,
               monster_dim,
               bullet_dim,
               monster_entity_num,
               move_output_size,
               16,
               16,
               128,
               128) {
}

ActorNet::ActorNet(int player_dim,
                   int monster_dim,
                   int bullet_dim,
                   int monster_entity_num,
                   int move_output_size,
                   int embedding_dim,
                   int attention_key_dim,
                   int gru_hidden_dim,
                   int out_hidden_dim)
    : m_player_dim(player_dim),
      m_monster_dim(monster_dim),
      m_bullet_dim(bullet_dim),
      m_monster_entity_num(monster_entity_num),
      m_move_output_size(move_output_size),
      m_embedding_dim(embedding_dim),
      m_attention_key_dim(attention_key_dim),
      m_gru_hidden_dim(gru_hidden_dim),
      m_out_hidden_dim(out_hidden_dim) {
    if (player_dim <= 0 || monster_dim <= 0 || bullet_dim <= 0 ||
        monster_entity_num <= 0 || move_output_size != 2 || embedding_dim <= 0 ||
        attention_key_dim <= 0 || gru_hidden_dim <= 0 ||
        out_hidden_dim <= 0) {
        MNN_ERROR("ActorNet：移动输出维度必须为2，其他网络维度必须为正整数。\n");
        return;
    }

    // 两个动作头完全独立，且输出层不附加激活。
    m_move_fc = make_parameter({m_out_hidden_dim, m_gru_hidden_dim});
    m_move_fc_bias = make_parameter({m_out_hidden_dim});
    m_move_fc2 = make_parameter({m_out_hidden_dim/2, m_out_hidden_dim});
    m_move_fc2_bias = make_parameter({m_out_hidden_dim/2});
    m_move_fc_out =
        make_parameter({m_move_output_size, m_out_hidden_dim/2});
    m_move_fc_out_bias = make_parameter({m_move_output_size});

    // 唯一的射击输出层复用主注意力表示，一次生成全部分数和开火logit。
    m_shoot_fc = make_parameter(
        {m_monster_entity_num + 1, 21 * m_embedding_dim});
    m_shoot_fc_bias = make_parameter({m_monster_entity_num + 1});

    addParameter(m_move_fc);
    addParameter(m_move_fc_bias);
    addParameter(m_move_fc2);
    addParameter(m_move_fc2_bias);
    addParameter(m_move_fc_out);
    addParameter(m_move_fc_out_bias);
    addParameter(m_shoot_fc);
    addParameter(m_shoot_fc_bias);

    m_player_embedding =
        std::make_shared<Embedding>(m_player_dim, m_embedding_dim);
    m_monster_embedding =
        std::make_shared<Embedding>(m_monster_dim, m_embedding_dim);
    m_bullet_embedding =
        std::make_shared<Embedding>(m_bullet_dim, m_embedding_dim);
    m_preprocess_attention =
        std::make_shared<Attention>(m_embedding_dim, m_attention_key_dim);
    // m_preprocess_state_pooling =
    //     std::make_shared<StatePooling>(m_embedding_dim);
    m_preprocess_gru =
        std::make_shared<GRU>(m_embedding_dim * 21, m_gru_hidden_dim);

    // AIAgent对每层统一执行N(0,0.1)初始化，因此覆盖子模块自身的默认初始化。
    initialize_normal(m_player_embedding);
    initialize_normal(m_monster_embedding);
    initialize_normal(m_bullet_embedding);
    initialize_normal(m_preprocess_attention);
    initialize_normal(m_preprocess_gru);
    registerModel({
        m_player_embedding,
        m_monster_embedding,
        m_bullet_embedding,
        m_preprocess_attention,
        // m_preprocess_state_pooling,
        m_preprocess_gru});
    setType("ActorNet");
}

std::vector<MNN::Express::VARP> ActorNet::onForward(
    const std::vector<MNN::Express::VARP>& inputs) {
    if (inputs.size() != 4 || inputs[0] == nullptr || inputs[1] == nullptr ||
        inputs[2] == nullptr || inputs[3] == nullptr) {
        MNN_ERROR("ActorNet: player, monster, bullet and mask are required.\n");
        return {};
    }
    const auto* player_info = inputs[0]->getInfo();
    const auto* monster_info = inputs[1]->getInfo();
    const auto* bullet_info = inputs[2]->getInfo();
    const auto valid_data = [](const MNN::Express::Variable::Info* info,
                               int feature_dim) {
        return info != nullptr && info->dim.size() == 3 && info->dim[0] > 0 &&
            info->dim[1] > 0 && info->dim[2] == feature_dim &&
            info->type == halide_type_of<float>();
    };
    if (!valid_data(player_info, m_player_dim) || player_info->dim[1] != 1 ||
        !valid_data(monster_info, m_monster_dim) ||
        monster_info->dim[1] != m_monster_entity_num ||
        !valid_data(bullet_info, m_bullet_dim) ||
        monster_info->dim[0] != player_info->dim[0] ||
        bullet_info->dim[0] != player_info->dim[0]) {
        MNN_ERROR("ActorNet: invalid player/monster/bullet tensor shapes.\n");
        return {};
    }
    const int entity_count = player_info->dim[1] + monster_info->dim[1] +
        bullet_info->dim[1];
    const auto* mask_info = inputs[3]->getInfo();
    if (entity_count != 21 || mask_info == nullptr || mask_info->dim.size() != 2 ||
        mask_info->dim[0] != player_info->dim[0] ||
        mask_info->dim[1] != entity_count ||
        mask_info->type != halide_type_of<float>()) {
        MNN_ERROR("ActorNet: mask must be a float32 tensor shaped [B,E].\n");
        return {};
    }

    using namespace MNN::Express;
    const int batch_size = player_info->dim[0];
    const auto* hidden_info =
        m_gru_hidden == nullptr ? nullptr : m_gru_hidden->getInfo();
    if (hidden_info == nullptr || hidden_info->dim.size() != 2 ||
        hidden_info->dim[0] != batch_size ||
        hidden_info->dim[1] != m_gru_hidden_dim) {
        std::vector<float> zeros(
            static_cast<std::size_t>(batch_size * m_gru_hidden_dim), 0.0f);
        m_gru_hidden =
            _Const(zeros.data(), {batch_size, m_gru_hidden_dim}, NCHW);
    }

    const auto player = _Relu(m_player_embedding->forward(inputs[0]));
    const auto monster = _Relu(m_monster_embedding->forward(inputs[1]));
    const auto bullet = _Relu(m_bullet_embedding->forward(inputs[2]));
    const auto embedded = _Concat({player, monster, bullet}, 1);
    const auto attended = m_preprocess_attention->forward(embedded, inputs[3]);
    // const auto pooled = m_preprocess_state_pooling->forward(attended);
    const auto flattened = _Reshape(attended, {batch_size, -1});
    const auto hidden_next =
        m_preprocess_gru->forwardStep(flattened, m_gru_hidden);

    const auto move_hidden1 = _Relu(
        _Add(_MatMul(hidden_next, m_move_fc, false, true),
             m_move_fc_bias));
    const auto move_hidden2 = _Relu(
        _Add(_MatMul(move_hidden1, m_move_fc2, false, true),
             m_move_fc2_bias));
    const auto raw_move_output =
        _Add(_MatMul(move_hidden2, m_move_fc_out, false, true),
             m_move_fc_out_bias);
    // 移动头只输出水平和垂直两个原始均值，采样与离散化由 AIAgent 完成。
    const auto move_output = raw_move_output;

    // 完整掩码布局为[player, monsters..., bullets...]，这里切出怪物段。
    const int monster_mask_starts[] = {0, 1};
    const int monster_mask_sizes[] = {batch_size, m_monster_entity_num};
    const auto monster_mask = _Slice(
        inputs[3], _Const(monster_mask_starts, {2}, NCHW, halide_type_of<int>()),
        _Const(monster_mask_sizes, {2}, NCHW, halide_type_of<int>()));
    const auto raw_shoot_output = _Add(
        _MatMul(flattened, m_shoot_fc, false, true), m_shoot_fc_bias);
    const int score_starts[] = {0, 0};
    const int score_sizes[] = {batch_size, m_monster_entity_num};
    auto shoot_scores = _Slice(
        raw_shoot_output, _Const(score_starts, {2}, NCHW, halide_type_of<int>()),
        _Const(score_sizes, {2}, NCHW, halide_type_of<int>()));
    // padding分数固定为极小有限值，避免它被目标选择逻辑选中。
    shoot_scores = shoot_scores +
        (_Scalar<float>(1.0f) - monster_mask) * _Scalar<float>(-1.0e9f);
    const int possibility_starts[] = {0, m_monster_entity_num};
    const int possibility_sizes[] = {batch_size, 1};
    const auto shoot_possibility = _Slice(
        raw_shoot_output,
        _Const(possibility_starts, {2}, NCHW, halide_type_of<int>()),
        _Const(possibility_sizes, {2}, NCHW, halide_type_of<int>()));
    const auto shoot_output = _Concat({shoot_scores, shoot_possibility}, 1);

    // 保存数值常量而不是计算表达式，使跨时间步行为与旧网络的截断记忆一致。
    const float* hidden_values = hidden_next->readMap<float>();
    if (hidden_values == nullptr) {
        MNN_ERROR("ActorNet：GRU隐藏状态读取失败。\n");
        return {};
    }
    std::vector<float> detached_hidden(
        hidden_values,
        hidden_values +
            static_cast<std::size_t>(batch_size * m_gru_hidden_dim));
    m_gru_hidden = _Const(
        detached_hidden.data(), {batch_size, m_gru_hidden_dim}, NCHW);

    // move和shoot保持原始动作值；hidden_next供Critic消费并保留当前步梯度图。
    return {move_output, shoot_output, hidden_next};
}

void ActorNet::ResetAllGRUMemory() {
    m_gru_hidden = nullptr;
}

void ActorNet::ResetGRUMemory(int batch_index) {
    if (m_gru_hidden == nullptr || m_gru_hidden->getInfo() == nullptr) {
        MNN_ERROR("ActorNet：GRU记忆尚未建立。\n");
        return;
    }
    const auto* info = m_gru_hidden->getInfo();
    if (batch_index < 0 || batch_index >= info->dim[0]) {
        MNN_ERROR("ActorNet：GRU记忆下标越界。\n");
        return;
    }

    const float* current = m_gru_hidden->readMap<float>();
    std::vector<float> values(current, current + info->size);
    const std::size_t row_begin =
        static_cast<std::size_t>(batch_index * m_gru_hidden_dim);
    std::fill(values.begin() + row_begin,
              values.begin() + row_begin + m_gru_hidden_dim,
              0.0f);
    m_gru_hidden = MNN::Express::_Const(
        values.data(), info->dim, MNN::Express::NCHW);
}

void ActorNet::CacheMemory()
{
    m_gru_hidden_cache = MNN::Express::_Clone(m_gru_hidden, true);
}

void ActorNet::RestoreMemory()
{
    m_gru_hidden = MNN::Express::_Clone(m_gru_hidden_cache, true);
}

MNN::Express::Module* ActorNet::clone(CloneContext* context) const {
    std::unique_ptr<ActorNet> module(new ActorNet);
    module->m_player_dim = m_player_dim;
    module->m_monster_dim = m_monster_dim;
    module->m_bullet_dim = m_bullet_dim;
    module->m_monster_entity_num = m_monster_entity_num;
    module->m_move_output_size = m_move_output_size;
    module->m_embedding_dim = m_embedding_dim;
    module->m_attention_key_dim = m_attention_key_dim;
    module->m_gru_hidden_dim = m_gru_hidden_dim;
    module->m_out_hidden_dim = m_out_hidden_dim;

    module->m_move_fc = m_move_fc;
    module->m_move_fc_bias = m_move_fc_bias;
    module->m_move_fc2 = m_move_fc2;
    module->m_move_fc2_bias = m_move_fc2_bias;
    module->m_move_fc_out = m_move_fc_out;
    module->m_move_fc_out_bias = m_move_fc_out_bias;
    module->m_shoot_fc = m_shoot_fc;
    module->m_shoot_fc_bias = m_shoot_fc_bias;

    module->addParameter(module->m_move_fc);
    module->addParameter(module->m_move_fc_bias);
    module->addParameter(module->m_move_fc2);
    module->addParameter(module->m_move_fc2_bias);
    module->addParameter(module->m_move_fc_out);
    module->addParameter(module->m_move_fc_out_bias);
    module->addParameter(module->m_shoot_fc);
    module->addParameter(module->m_shoot_fc_bias);

    module->m_player_embedding = clone_child(m_player_embedding, context);
    module->m_monster_embedding = clone_child(m_monster_embedding, context);
    module->m_bullet_embedding = clone_child(m_bullet_embedding, context);
    module->m_preprocess_attention =
        clone_child(m_preprocess_attention, context);
    module->m_preprocess_gru =
        clone_child(m_preprocess_gru, context);
    if (module->m_player_embedding == nullptr ||
        module->m_monster_embedding == nullptr ||
        module->m_bullet_embedding == nullptr ||
        module->m_preprocess_attention == nullptr ||
        module->m_preprocess_gru == nullptr) {
        return nullptr;
    }
    module->registerModel({
        module->m_player_embedding,
        module->m_monster_embedding,
        module->m_bullet_embedding,
        module->m_preprocess_attention,
        module->m_preprocess_gru});
    auto* result = cloneBaseTo(context, module.get());
    module.release();
    return result;
}

} // MiniMind命名空间
