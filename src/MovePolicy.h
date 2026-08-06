#pragma once

#include <MNN/expr/ExprCreator.hpp>
#include <MNN/expr/NeuralNetWorkOp.hpp>

#include <algorithm>
#include <array>
#include <cmath>
#include <vector>

namespace MiniMind::MovePolicy {

// 移动策略的水平轴和垂直轴统一使用此温度，后续调参只需修改这一处。
inline constexpr float kMoveTemperature = 0.5f;
static_assert(kMoveTemperature > 0.0f, "移动温度必须大于0");

inline std::array<double, 3> StableWeights(const float *logits, int offset) {
    std::array<double, 3> scaled_logits{};
    for (int index = 0; index < 3; ++index) {
        scaled_logits[index] =
            static_cast<double>(logits[offset + index]) / kMoveTemperature;
    }
    const double maximum = *std::max_element(
        scaled_logits.begin(), scaled_logits.end());
    std::array<double, 3> weights{};
    for (int index = 0; index < 3; ++index) {
        weights[index] = std::exp(scaled_logits[index] - maximum);
    }
    return weights;
}

inline MNN::Express::VARP AxisLogProbability(
    const MNN::Express::VARP &actions,
    const MNN::Express::VARP &move_logits,
    int move_offset, int action_column, int batch_size) {
    using namespace MNN::Express;

    // 温度除法保留在Express计算图中，使原始logits的梯度自然包含1/T。
    const auto axis_logits = _Split(move_logits, {3, 3}, 1)[move_offset / 3];
    const auto tempered_logits =
        axis_logits / _Scalar<float>(kMoveTemperature);
    const auto maximum = _ReduceMax(tempered_logits, {1}, true);
    const auto log_normalizer = maximum + _Log(
        _ReduceSum(_Exp(tempered_logits - maximum), {1}, true));
    const int action_width = actions->getInfo()->dim[1];
    const float *action_values = actions->readMap<float>();
    std::vector<float> one_hot_values(
        static_cast<std::size_t>(batch_size) * 3, 0.0f);
    for (int batch = 0; batch < batch_size; ++batch) {
        const int action = static_cast<int>(
            action_values[batch * action_width + action_column]);
        if (action >= 0 && action < 3) {
            one_hot_values[batch * 3 + action] = 1.0f;
        }
    }
    // 采样动作是固定训练标签，只让logits参与策略梯度计算。
    const auto one_hot = _Const(
        one_hot_values.data(), {batch_size, 3}, NCHW);
    return _ReduceSum(
        (tempered_logits - log_normalizer) * one_hot, {1}, true);
}

inline MNN::Express::VARP JointLogProbability(
    const MNN::Express::VARP &actions,
    const MNN::Express::VARP &move_logits) {
    if (actions == nullptr || move_logits == nullptr ||
        actions->getInfo() == nullptr || move_logits->getInfo() == nullptr ||
        actions->getInfo()->dim.size() != 2 ||
        move_logits->getInfo()->dim.size() != 2 ||
        actions->getInfo()->dim[0] != move_logits->getInfo()->dim[0] ||
        actions->getInfo()->dim[1] < 2 ||
        move_logits->getInfo()->dim[1] < 6) {
        return nullptr;
    }
    const int batch_size = actions->getInfo()->dim[0];
    return AxisLogProbability(actions, move_logits, 0, 0, batch_size)
        + AxisLogProbability(actions, move_logits, 3, 1, batch_size);
}

} // MiniMind::MovePolicy命名空间
