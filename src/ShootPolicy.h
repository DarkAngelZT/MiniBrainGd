#pragma once

#include <MNN/expr/ExprCreator.hpp>
#include <MNN/expr/MathOp.hpp>
#include <MNN/expr/NeuralNetWorkOp.hpp>

#include <algorithm>
#include <cmath>
#include <limits>
#include <vector>

namespace MiniMind::ShootPolicy {

constexpr float kMaskedScore = -1.0e9f;

inline int MaskedArgmax(const float* scores, const float* mask, int count) {
    int selected = 0;
    float best = -std::numeric_limits<float>::infinity();
    bool has_valid = false;
    for (int index = 0; index < count; ++index) {
        if (mask[index] > 0.0f && (!has_valid || scores[index] > best)) {
            selected = index;
            best = scores[index];
            has_valid = true;
        }
    }
    return selected;
}

inline int SampleCategorical(const float* scores, const float* mask,
                             int count, float unit_random) {
    float maximum = -std::numeric_limits<float>::infinity();
    bool has_valid = false;
    for (int index = 0; index < count; ++index) {
        if (mask[index] > 0.0f) {
            maximum = std::max(maximum, scores[index]);
            has_valid = true;
        }
    }
    if (!has_valid) {
        return 0;
    }
    std::vector<float> weights(static_cast<std::size_t>(count), 0.0f);
    float total = 0.0f;
    for (int index = 0; index < count; ++index) {
        if (mask[index] > 0.0f) {
            weights[index] = std::exp(scores[index] - maximum);
            total += weights[index];
        }
    }
    const float threshold = std::clamp(unit_random, 0.0f, 0.99999994f) * total;
    float cumulative = 0.0f;
    int last_valid = 0;
    for (int index = 0; index < count; ++index) {
        if (mask[index] <= 0.0f) {
            continue;
        }
        last_valid = index;
        cumulative += weights[index];
        if (threshold < cumulative) {
            return index;
        }
    }
    return last_valid;
}

inline MNN::Express::VARP SelectColumns(
    const MNN::Express::VARP& value, int begin, int count) {
    using namespace MNN::Express;
    const int width = value->getInfo()->dim[1];
    std::vector<float> selector(static_cast<std::size_t>(width * count), 0.0f);
    for (int index = 0; index < count; ++index) {
        selector[static_cast<std::size_t>((begin + index) * count + index)] = 1.0f;
    }
    return _MatMul(value, _Const(selector.data(), {width, count}, NCHW));
}

inline MNN::Express::VARP LogSigmoid(const MNN::Express::VARP& value) {
    using namespace MNN::Express;
    const auto zero = _Scalar<float>(0.0f);
    return _Minimum(value, zero) - _Log1p(_Exp(_Negative(_Abs(value))));
}

inline MNN::Express::VARP LogProbability(
    const MNN::Express::VARP& actions,
    const MNN::Express::VARP& shoot_output,
    const MNN::Express::VARP& monster_mask) {
    using namespace MNN::Express;
    const int monster_count = monster_mask->getInfo()->dim[1];
    const auto scores = SelectColumns(shoot_output, 0, monster_count);
    const auto shoot_logit = SelectColumns(shoot_output, monster_count, 1);
    const auto masked_scores = scores +
        (_Scalar<float>(1.0f) - monster_mask) * _Scalar<float>(kMaskedScore);
    const auto maximum = _ReduceMax(masked_scores, {1}, true);
    const auto log_normalizer = maximum +
        _Log(_ReduceSum(_Exp(masked_scores - maximum), {1}, true));
    const auto target_index = _Reshape(
        _Cast(SelectColumns(actions, 2, 1), halide_type_of<int>()), {-1});
    const auto target_one_hot = _OneHot(
        target_index, _Scalar<int>(monster_count), _Scalar<float>(1.0f),
        _Scalar<float>(0.0f));
    const auto selected_mask = _ReduceSum(target_one_hot * monster_mask, {1}, true);
    const auto has_valid = _Minimum(
        _ReduceSum(monster_mask, {1}, true), _Scalar<float>(1.0f));
    const auto target_log_probability = _ReduceSum(
        (masked_scores - log_normalizer) * target_one_hot, {1}, true)
        * selected_mask * has_valid;

    // 训练缓存保存连续开火概率；旧策略口径只累计“选择开火”的对数概率。
    const auto shoot_log_probability = LogSigmoid(shoot_logit);
    return target_log_probability + shoot_log_probability;
}

} // MiniMind::ShootPolicy 命名空间
