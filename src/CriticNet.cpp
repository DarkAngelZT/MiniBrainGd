#include "CriticNet.h"

#include <MNN/expr/ExprCreator.hpp>
#include <MNN/expr/NeuralNetWorkOp.hpp>
#include <MNN/MNNDefine.h>

#include <random>
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

} // 匿名命名空间

CriticNet::CriticNet(int input_size, int hidden_size)
    : m_input_size(input_size), m_hidden_size(hidden_size) {
    if (input_size <= 0 || hidden_size <= 0) {
        MNN_ERROR("CriticNet：input_size和hidden_size必须为正整数。\n");
        return;
    }

    // 参数顺序严格对应原网络的第一层权重、第一层偏置、输出层权重、输出层偏置。
    m_fc = make_parameter({m_hidden_size, m_input_size});
    m_fc_bias = make_parameter({m_hidden_size});
    m_fc_out = make_parameter({1, m_hidden_size});
    m_fc_out_bias = make_parameter({1});
    addParameter(m_fc);
    addParameter(m_fc_bias);
    addParameter(m_fc_out);
    addParameter(m_fc_out_bias);
    setType("CriticNet");
}

std::vector<MNN::Express::VARP> CriticNet::onForward(
    const std::vector<MNN::Express::VARP>& inputs) {
    if (inputs.size() != 1 || inputs[0] == nullptr) {
        MNN_ERROR("CriticNet：必须提供一个非空输入。\n");
        return {};
    }
    const auto* info = inputs[0]->getInfo();
    if (info == nullptr || info->dim.size() != 2 || info->dim[0] <= 0 ||
        info->dim[1] != m_input_size ||
        info->type != halide_type_of<float>()) {
        MNN_ERROR("CriticNet：输入形状必须为[batch,input_size]的float32张量。\n");
        return {};
    }

    using namespace MNN::Express;
    const auto hidden =
        _Relu(_Add(_MatMul(inputs[0], m_fc, false, true), m_fc_bias));
    return {_Add(_MatMul(hidden, m_fc_out, false, true), m_fc_out_bias)};
}

MNN::Express::Module* CriticNet::clone(CloneContext* context) const {
    auto* module = new CriticNet;
    module->m_input_size = m_input_size;
    module->m_hidden_size = m_hidden_size;
    module->m_fc = m_fc;
    module->m_fc_bias = m_fc_bias;
    module->m_fc_out = m_fc_out;
    module->m_fc_out_bias = m_fc_out_bias;
    module->addParameter(module->m_fc);
    module->addParameter(module->m_fc_bias);
    module->addParameter(module->m_fc_out);
    module->addParameter(module->m_fc_out_bias);
    return cloneBaseTo(context, module);
}

} // MiniMind命名空间
