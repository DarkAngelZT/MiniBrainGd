#pragma once

#include <MNN/expr/ExprCreator.hpp>
#include <MNN/expr/MathOp.hpp>

#include <algorithm>
#include <array>
#include <cmath>
#include <vector>

namespace MiniMind::MovePolicy {

constexpr float kProbabilityEpsilon = 1.0e-6f;
constexpr float kInitialLogisticScale = 1.0f;
constexpr float kMinimumLogisticScale = 0.15f;
constexpr float kLogisticScaleDecay = 0.995f;
constexpr float kLeftProbabilityBoundary = 1.0f / 3.0f;
constexpr float kRightProbabilityBoundary = 2.0f / 3.0f;
constexpr float kLeftLogitBoundary = -0.6931471805599453f;
constexpr float kRightLogitBoundary = 0.6931471805599453f;

inline float Sigmoid(float value) {
    return 1.0f / (1.0f + std::exp(-value));
}

inline float DiscretizeProbability(float probability) {
    if (probability <= kLeftProbabilityBoundary) {
        return -1.0f;
    }
    if (probability >= kRightProbabilityBoundary) {
        return 1.0f;
    }
    return 0.0f;
}

inline float DeterministicAction(float mean) {
    return DiscretizeProbability(Sigmoid(mean));
}

inline float SampleAction(float mean, float scale, float uniform) {
    const float bounded_uniform = std::clamp(
        uniform, kProbabilityEpsilon, 1.0f - kProbabilityEpsilon);
    const float logistic_noise = scale *
        (std::log(bounded_uniform) - std::log(1.0f - bounded_uniform));
    return DiscretizeProbability(Sigmoid(mean + logistic_noise));
}

inline std::array<float, 3> AxisProbabilities(float mean, float scale) {
    const float left = Sigmoid((kLeftLogitBoundary - mean) / scale);
    const float right_cdf = Sigmoid((kRightLogitBoundary - mean) / scale);
    return {left, right_cdf - left, 1.0f - right_cdf};
}

inline float DecayScale(float scale) {
    return std::max(kMinimumLogisticScale, scale * kLogisticScaleDecay);
}

inline MNN::Express::VARP SliceColumn(
    const MNN::Express::VARP &value, int row_count, int column) {
    (void)row_count;
    const int column_count = value->getInfo()->dim[1];
    std::vector<float> selector(static_cast<std::size_t>(column_count), 0.0f);
    selector[static_cast<std::size_t>(column)] = 1.0f;
    const auto selector_tensor = MNN::Express::_Const(
        selector.data(), {1, column_count}, MNN::Express::NCHW);
    // 乘选择向量再求和，保持切片操作的梯度。
    return MNN::Express::_ReduceSum(
        value * selector_tensor, {1}, true);
}

inline MNN::Express::VARP LogisticLogProbability(
    const MNN::Express::VARP &actions,
    const MNN::Express::VARP &move_means,
    float scale) {
    using namespace MNN::Express;
    const int batch_size = actions->getInfo()->dim[0];
    const auto scale_value = _Scalar<float>(scale);
    const auto epsilon = _Scalar<float>(kProbabilityEpsilon);

    auto axis_log_probability = [&](int action_column, int mean_column) {
        const auto action = SliceColumn(actions, batch_size, action_column);
        const auto mean = SliceColumn(move_means, batch_size, mean_column);
        const auto left = _Sigmoid(
            (_Scalar<float>(kLeftLogitBoundary) - mean) / scale_value);
        const auto right_cdf = _Sigmoid(
            (_Scalar<float>(kRightLogitBoundary) - mean) / scale_value);
        const auto center = right_cdf - left;
        const auto right = _Scalar<float>(1.0f) - right_cdf;

        // 动作只取 -1、0、1，下面三个多项式正好是对应区间的选择器。
        const auto select_left =
            _Scalar<float>(0.5f) * action * (action - _Scalar<float>(1.0f));
        const auto select_center = _Scalar<float>(1.0f) - _Square(action);
        const auto select_right =
            _Scalar<float>(0.5f) * action * (action + _Scalar<float>(1.0f));
        const auto selected_probability =
            select_left * left + select_center * center + select_right * right;
        return _Log(selected_probability + epsilon);
    };

    return axis_log_probability(0, 0) + axis_log_probability(1, 1);
}

} // MiniMind::MovePolicy 命名空间
