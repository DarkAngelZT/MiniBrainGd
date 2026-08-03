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

ActorNet::ActorNet(int input_size,
                   int move_output_size,
                   int shoot_output_size)
    : ActorNet(input_size,
               move_output_size,
               shoot_output_size,
               16,
               16,
               128,
               128) {
}

ActorNet::ActorNet(int entity_feature_dim,
                   int move_output_size,
                   int shoot_output_size,
                   int embedding_dim,
                   int attention_key_dim,
                   int gru_hidden_dim,
                   int out_hidden_dim)
    : m_input_size(entity_feature_dim),
      m_move_output_size(move_output_size),
      m_shoot_output_size(shoot_output_size),
      m_embedding_dim(embedding_dim),
      m_attention_key_dim(attention_key_dim),
      m_gru_hidden_dim(gru_hidden_dim),
      m_out_hidden_dim(out_hidden_dim) {
    if (entity_feature_dim <= 0 || move_output_size <= 0 ||
        shoot_output_size <= 0 || embedding_dim <= 0 ||
        attention_key_dim <= 0 || gru_hidden_dim <= 0 ||
        out_hidden_dim <= 0) {
        MNN_ERROR("ActorNet：所有网络维度必须为正整数。\n");
        return;
    }

    // 两个动作头完全独立，且输出层不附加激活。
    m_move_fc = make_parameter({m_out_hidden_dim, m_gru_hidden_dim});
    m_move_fc_bias = make_parameter({m_out_hidden_dim});
    m_move_fc_out =
        make_parameter({m_move_output_size, m_out_hidden_dim});
    m_move_fc_out_bias = make_parameter({m_move_output_size});

    m_shoot_fc = make_parameter({m_out_hidden_dim, m_gru_hidden_dim});
    m_shoot_fc_bias = make_parameter({m_out_hidden_dim});
    m_shoot_fc_out =
        make_parameter({m_shoot_output_size, m_out_hidden_dim});
    m_shoot_fc_out_bias = make_parameter({m_shoot_output_size});

    addParameter(m_move_fc);
    addParameter(m_move_fc_bias);
    addParameter(m_move_fc_out);
    addParameter(m_move_fc_out_bias);
    addParameter(m_shoot_fc);
    addParameter(m_shoot_fc_bias);
    addParameter(m_shoot_fc_out);
    addParameter(m_shoot_fc_out_bias);

    m_preprocess_embedding =
        std::make_shared<Embedding>(m_input_size, m_embedding_dim);
    m_preprocess_attention =
        std::make_shared<Attention>(m_embedding_dim, m_attention_key_dim);
    // m_preprocess_state_pooling =
    //     std::make_shared<StatePooling>(m_embedding_dim);
    m_preprocess_gru =
        std::make_shared<GRU>(m_embedding_dim * 21, m_gru_hidden_dim);

    // AIAgent对每层统一执行N(0,0.1)初始化，因此覆盖子模块自身的默认初始化。
    initialize_normal(m_preprocess_embedding);
    initialize_normal(m_preprocess_attention);
    initialize_normal(m_preprocess_gru);
    registerModel({
        m_preprocess_embedding,
        m_preprocess_attention,
        // m_preprocess_state_pooling,
        m_preprocess_gru});
    setType("ActorNet");
}

std::vector<MNN::Express::VARP> ActorNet::onForward(
    const std::vector<MNN::Express::VARP>& inputs) {
    if (inputs.size() != 1 || inputs[0] == nullptr) {
        MNN_ERROR("ActorNet：必须提供一个非空state输入。\n");
        return {};
    }
    const auto* state_info = inputs[0]->getInfo();
    if (state_info == nullptr || state_info->dim.size() != 3 ||
        state_info->dim[0] <= 0 ||
        state_info->dim[1] <= 0 ||
        state_info->dim[2] != m_input_size ||
        state_info->type != halide_type_of<float>()) {
        MNN_ERROR("ActorNet：state必须为[B,E,F]的float32张量。\n");
        return {};
    }

    using namespace MNN::Express;
    const int batch_size = state_info->dim[0];
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

    const auto embedded = _Relu(m_preprocess_embedding->forward(inputs[0]));
    const auto attended = m_preprocess_attention->forward(embedded);
    // const auto pooled = m_preprocess_state_pooling->forward(attended);
    const auto flattened = _Reshape(attended, {batch_size, -1});
    const auto hidden_next =
        m_preprocess_gru->forwardStep(flattened, m_gru_hidden);

    const auto move_hidden = _Relu(
        _Add(_MatMul(hidden_next, m_move_fc, false, true),
             m_move_fc_bias));
    const auto move_output =
        _Add(_MatMul(move_hidden, m_move_fc_out, false, true),
             m_move_fc_out_bias);

    const auto shoot_hidden = _Relu(
        _Add(_MatMul(hidden_next, m_shoot_fc, false, true),
             m_shoot_fc_bias));
    const auto shoot_output =
        _Add(_MatMul(shoot_hidden, m_shoot_fc_out, false, true),
             m_shoot_fc_out_bias);

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
    module->m_input_size = m_input_size;
    module->m_move_output_size = m_move_output_size;
    module->m_shoot_output_size = m_shoot_output_size;
    module->m_embedding_dim = m_embedding_dim;
    module->m_attention_key_dim = m_attention_key_dim;
    module->m_gru_hidden_dim = m_gru_hidden_dim;
    module->m_out_hidden_dim = m_out_hidden_dim;

    module->m_move_fc = m_move_fc;
    module->m_move_fc_bias = m_move_fc_bias;
    module->m_move_fc_out = m_move_fc_out;
    module->m_move_fc_out_bias = m_move_fc_out_bias;
    module->m_shoot_fc = m_shoot_fc;
    module->m_shoot_fc_bias = m_shoot_fc_bias;
    module->m_shoot_fc_out = m_shoot_fc_out;
    module->m_shoot_fc_out_bias = m_shoot_fc_out_bias;

    module->addParameter(module->m_move_fc);
    module->addParameter(module->m_move_fc_bias);
    module->addParameter(module->m_move_fc_out);
    module->addParameter(module->m_move_fc_out_bias);
    module->addParameter(module->m_shoot_fc);
    module->addParameter(module->m_shoot_fc_bias);
    module->addParameter(module->m_shoot_fc_out);
    module->addParameter(module->m_shoot_fc_out_bias);

    module->m_preprocess_embedding =
        clone_child(m_preprocess_embedding, context);
    module->m_preprocess_attention =
        clone_child(m_preprocess_attention, context);
    module->m_preprocess_state_pooling =
        clone_child(m_preprocess_state_pooling, context);
    module->m_preprocess_gru =
        clone_child(m_preprocess_gru, context);
    if (module->m_preprocess_embedding == nullptr ||
        module->m_preprocess_attention == nullptr ||
        module->m_preprocess_state_pooling == nullptr ||
        module->m_preprocess_gru == nullptr) {
        return nullptr;
    }
    module->registerModel({
        module->m_preprocess_embedding,
        module->m_preprocess_attention,
        module->m_preprocess_state_pooling,
        module->m_preprocess_gru});
    auto* result = cloneBaseTo(context, module.get());
    module.release();
    return result;
}

} // MiniMind命名空间
