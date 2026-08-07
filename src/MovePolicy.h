#pragma once

#include <MNN/expr/ExprCreator.hpp>
#include <MNN/expr/MathOp.hpp>
#include <MNN/expr/NeuralNetWorkOp.hpp>

#include <algorithm>
#include <cmath>
#include <vector>

namespace MiniMind::MovePolicy {

constexpr float kSigmaEpsilon = 1.0e-6f;
constexpr float kMaximumSigma = 1.0e6f;
constexpr float kHalfLogTwoPi = 0.9189385332046727f;

inline float DiscretizeRawAction(float action) {
    if (action <= -0.5f) {
        return -1.0f;
    }
    if (action >= 0.5f) {
        return 1.0f;
    }
    return 0.0f;
}

inline float Sample(float mean, float sigma, float standard_normal) {
    return mean + sigma * standard_normal;
}

inline float DeterministicMean(float mean) {
    return mean;
}

inline MNN::Express::VARP PositiveSigma(
    const MNN::Express::VARP &raw_sigma) {
    return MNN::Express::_Softplus(raw_sigma)
        + MNN::Express::_Scalar<float>(kSigmaEpsilon);
}

inline MNN::Express::VARP SliceColumn(
    const MNN::Express::VARP &value, int row_count, int column) {
    (void)row_count;
    const int column_count = value->getInfo()->dim[1];
    std::vector<float> selector(static_cast<std::size_t>(column_count), 0.0f);
    selector[static_cast<std::size_t>(column)] = 1.0f;
    const auto selector_tensor = MNN::Express::_Const(
        selector.data(), {1, column_count}, MNN::Express::NCHW);
    // 乘选择向量再求和，避免 MNN Slice 节点截断参数梯度。
    return MNN::Express::_ReduceSum(
        value * selector_tensor, {1}, true);
}

inline MNN::Express::VARP GaussianLogProbability(
    const MNN::Express::VARP &actions,
    const MNN::Express::VARP &move_parameters) {
    using namespace MNN::Express;
    const int batch_size = actions->getInfo()->dim[0];
    const auto minimum_sigma = _Scalar<float>(kSigmaEpsilon);
    const auto maximum_sigma = _Scalar<float>(kMaximumSigma);

    auto axis_log_probability = [&](int action_column, int mean_column,
                                    int sigma_column) {
        const auto action = SliceColumn(actions, batch_size, action_column);
        const auto mean = SliceColumn(
            move_parameters, batch_size, mean_column);
        const auto sigma = _Maximum(
            _Minimum(SliceColumn(move_parameters, batch_size, sigma_column),
                     maximum_sigma),
            minimum_sigma);
        const auto normalized = (action - mean) / sigma;
        return
            _Scalar<float>(-0.5f) * _Square(normalized)
            - _Log(sigma) - _Scalar<float>(kHalfLogTwoPi);
    };

    // 移动头布局固定为：[水平均值、水平标准差、垂直均值、垂直标准差]。
    return axis_log_probability(0, 0, 1)
        + axis_log_probability(1, 2, 3);
}

} // MiniMind::MovePolicy 命名空间
