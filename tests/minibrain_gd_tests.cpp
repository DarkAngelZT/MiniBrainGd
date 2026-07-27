#include <algorithm>
#include <cmath>
#include <iostream>
#include <memory>
#include <random>
#include <string>
#include <utility>
#include <vector>

#include "EmbeddingLayer.h"
#include "StatePooling.h"
#include "../MiniBrain/Source/Activations/ReLU.h"
#include "../MiniBrain/Source/Layers/Attention.h"
#include "../MiniBrain/Source/Layers/FullyConnected.h"
#include "../MiniBrain/Source/Layers/GRU.h"
#include "../MiniBrain/Source/LossFunc/RegressionMSE.h"
#include "../MiniBrain/Source/Network.h"
#include "../MiniBrain/Source/Optimizer/Adam.h"

namespace {

class TestSuite {
public:
    void expect_true(bool actual, const std::string &message) {
        if (actual) {
            ++passed_;
            return;
        }
        ++failed_;
        std::cerr << "FAIL: " << message << "\n";
    }

    void expect_equal(int actual, int expected, const std::string &message) {
        if (actual == expected) {
            ++passed_;
            return;
        }
        ++failed_;
        std::cerr << "FAIL: " << message << " (expected " << expected
                  << ", got " << actual << ")\n";
    }

    void expect_near(float actual, float expected, const std::string &message) {
        constexpr float tolerance = 1.0e-5f;
        if (std::fabs(actual - expected) <= tolerance) {
            ++passed_;
            return;
        }
        ++failed_;
        std::cerr << "FAIL: " << message << " (expected " << expected
                  << ", got " << actual << ")\n";
    }

    int finish() const {
        std::cout << "Assertions: " << passed_ << " passed, " << failed_ << " failed\n";
        return failed_ == 0 ? 0 : 1;
    }

private:
    int passed_ = 0;
    int failed_ = 0;
};

class RecordingOptimizer : public MiniBrain::Optimizer<MiniBrain::Scalar> {
public:
    void Update(
        MiniBrain::ConstAlignedMapVec<MiniBrain::Scalar> &gradient,
        MiniBrain::AlignedMapVec<MiniBrain::Scalar> &,
        const void *) override {
        updates.emplace_back(gradient.data(), gradient.data() + gradient.size());
    }

    std::vector<std::vector<MiniBrain::Scalar>> updates;
};

void test_scalar_pooling_uses_each_batch_column_independently(TestSuite &suite) {
    MiniBrain::Matrix<MiniBrain::Scalar> input(4, 2);
    input.col(0) << 1.0f, 4.0f, 3.0f, 2.0f;
    input.col(1) << -1.0f, 6.0f, 5.0f, 0.0f;

    MiniBrain::StatePooling<MiniBrain::Scalar> pooling(4, 2);
    const auto output = pooling.Forward(input);

    suite.expect_equal(output.rows(), 4, "scalar pooling output row count");
    suite.expect_equal(output.cols(), 2, "scalar pooling preserves batch size");

    suite.expect_near(output(0, 0), 2.0f, "batch 0 feature 0 mean");
    suite.expect_near(output(1, 0), 3.0f, "batch 0 feature 1 mean");
    suite.expect_near(output(2, 0), 3.0f, "batch 0 feature 0 max");
    suite.expect_near(output(3, 0), 4.0f, "batch 0 feature 1 max");

    suite.expect_near(output(0, 1), 2.0f, "batch 1 feature 0 mean");
    suite.expect_near(output(1, 1), 3.0f, "batch 1 feature 1 mean");
    suite.expect_near(output(2, 1), 5.0f, "batch 1 feature 0 max");
    suite.expect_near(output(3, 1), 6.0f, "batch 1 feature 1 max");
}

void test_autodiff_pooling_preserves_values_and_expressions(TestSuite &suite) {
    MiniBrain::Matrix<MiniBrain::AutoDiffVar> input(4, 1);
    input(0, 0) = 2.0f;
    input(1, 0) = -3.0f;
    input(2, 0) = 6.0f;
    input(3, 0) = 5.0f;

    MiniBrain::StatePooling<MiniBrain::AutoDiffVar> pooling(4, 2);
    const auto output = pooling.Forward(input);

    suite.expect_equal(output.rows(), 4, "autodiff pooling output row count");
    suite.expect_equal(output.cols(), 1, "autodiff pooling preserves batch size");
    suite.expect_near(output(0, 0).expr->val, 4.0f, "autodiff feature 0 mean");
    suite.expect_near(output(1, 0).expr->val, 1.0f, "autodiff feature 1 mean");
    suite.expect_near(output(2, 0).expr->val, 6.0f, "autodiff feature 0 max");
    suite.expect_near(output(3, 0).expr->val, 5.0f, "autodiff feature 1 max");
}

void test_embedding_backward_aggregates_shared_entity_gradients(TestSuite &suite) {
    MiniBrain::EmbeddingLayer<MiniBrain::AutoDiffVar> embedding(4, 2, 1);
    MiniBrain::Random random;
    static_cast<MiniBrain::Layer<MiniBrain::AutoDiffVar> &>(embedding).Init(0.0f, 0.1f, random);
    embedding.SetParameters({1.0f, 2.0f, 0.0f});

    MiniBrain::Matrix<MiniBrain::AutoDiffVar> input(4, 1);
    input.col(0) << 1.0f, 3.0f, 2.0f, 4.0f;
    const auto output = embedding.Forward(input);
    MiniBrain::AutoDiffVar loss = output.sum();

    embedding.Backward(loss);
    RecordingOptimizer optimizer;
    embedding.Update(optimizer);

    suite.expect_equal(static_cast<int>(optimizer.updates.size()), 2, "embedding emits weight and bias gradients");
    suite.expect_equal(static_cast<int>(optimizer.updates[0].size()), 2, "embedding weight gradient size");
    suite.expect_near(optimizer.updates[0][0], 3.0f, "shared embedding weight 0 gradient sum");
    suite.expect_near(optimizer.updates[0][1], 7.0f, "shared embedding weight 1 gradient sum");
    suite.expect_near(optimizer.updates[1][0], 2.0f, "shared embedding bias gradient sum");
}

template <typename LayerType, typename... Args>
LayerType *add_initialized_layer(
    MiniBrain::Network<MiniBrain::AutoDiffVar> &network,
    MiniBrain::Random &random,
    Args &&...args) {
    auto layer = std::make_unique<LayerType>(std::forward<Args>(args)...);
    LayerType *layer_ptr = layer.get();
    static_cast<MiniBrain::Layer<MiniBrain::AutoDiffVar> *>(layer_ptr)->Init(0.0f, 0.1f, random);
    network.AddLayer(std::move(layer));
    return layer_ptr;
}

void calculate_log_probs_like_aiagent(
    const MiniBrain::Matrix<MiniBrain::Scalar> &action_new,
    const MiniBrain::Matrix<MiniBrain::AutoDiffVar> &move_data,
    const MiniBrain::Matrix<MiniBrain::AutoDiffVar> &shoot_data,
    MiniBrain::Matrix<MiniBrain::AutoDiffVar> &log_probs) {
    using autodiff::reverse::detail::abs;
    using autodiff::reverse::detail::atan2;
    using autodiff::reverse::detail::cos;
    using autodiff::reverse::detail::exp;
    using autodiff::reverse::detail::log;
    using autodiff::reverse::detail::sin;
    using autodiff::reverse::detail::sqrt;

    const int batch_size = action_new.cols();
    log_probs.resize(1, batch_size);

    for (int i = 0; i < batch_size; ++i) {
        MiniBrain::AutoDiffVar log_p_horizon = 0.0f;
        MiniBrain::AutoDiffVar log_p_vertical = 0.0f;

        const MiniBrain::Scalar act_horiz = action_new(i, 0);
        const MiniBrain::Scalar act_vert = action_new(i, 1);

        const MiniBrain::AutoDiffVar h0 = move_data(0, i);
        const MiniBrain::AutoDiffVar h1 = move_data(1, i);
        const MiniBrain::AutoDiffVar h2 = move_data(2, i);
        const MiniBrain::Scalar hmax = std::max({h0.expr->val, h1.expr->val, h2.expr->val});
        const MiniBrain::AutoDiffVar hlog_z = hmax + log(exp(h0 - hmax) + exp(h1 - hmax) + exp(h2 - hmax));
        log_p_horizon = act_horiz == 0.0f ? h0 - hlog_z : (act_horiz == 1.0f ? h1 - hlog_z : h2 - hlog_z);

        const MiniBrain::AutoDiffVar v0 = move_data(3, i);
        const MiniBrain::AutoDiffVar v1 = move_data(4, i);
        const MiniBrain::AutoDiffVar v2 = move_data(5, i);
        const MiniBrain::Scalar vmax = std::max({v0.expr->val, v1.expr->val, v2.expr->val});
        const MiniBrain::AutoDiffVar vlog_z = vmax + log(exp(v0 - vmax) + exp(v1 - vmax) + exp(v2 - vmax));
        log_p_vertical = act_vert == 0.0f ? v0 - vlog_z : (act_vert == 1.0f ? v1 - vlog_z : v2 - vlog_z);

        const MiniBrain::AutoDiffVar angle_x = shoot_data(0, i);
        const MiniBrain::AutoDiffVar angle_y = shoot_data(1, i);
        const MiniBrain::AutoDiffVar variance = shoot_data(2, i);
        const MiniBrain::AutoDiffVar std_dev = sqrt(abs(variance) + 1.0e-6f);
        const MiniBrain::AutoDiffVar angle = atan2(angle_y, angle_x);
        const MiniBrain::Scalar action_angle = std::atan2(action_new(i, 3), action_new(i, 2));
        MiniBrain::AutoDiffVar delta_theta = MiniBrain::AutoDiffVar(action_angle) - angle;
        delta_theta = atan2(sin(delta_theta), cos(delta_theta));
        const MiniBrain::AutoDiffVar normalized_diff = delta_theta / std_dev;
        constexpr MiniBrain::Scalar half_log_two_pi = 0.9189385332046727f;
        const MiniBrain::AutoDiffVar log_p_angle =
            -0.5f * normalized_diff * normalized_diff - log(std_dev) - half_log_two_pi;

        log_probs(0, i) = log_p_horizon + log_p_vertical + log_p_angle;
    }
}

bool parameters_changed(
    const std::vector<std::vector<MiniBrain::Scalar>> &before,
    const std::vector<std::vector<MiniBrain::Scalar>> &after) {
    if (before.size() != after.size()) {
        return true;
    }
    for (std::size_t layer = 0; layer < before.size(); ++layer) {
        if (before[layer].size() != after[layer].size()) {
            return true;
        }
        for (std::size_t parameter = 0; parameter < before[layer].size(); ++parameter) {
            if (std::fabs(before[layer][parameter] - after[layer][parameter]) > 1.0e-8f) {
                return true;
            }
        }
    }
    return false;
}

void test_aiagent_network_runs_two_training_epochs(TestSuite &suite) {
    constexpr int entity_feature_dim = 2;
    constexpr int entity_count = 3;
    constexpr int input_dim = entity_feature_dim * entity_count;
    constexpr int embedding_dim = 2;
    constexpr int attention_key_dim = 2;
    constexpr int gru_hidden_dim = embedding_dim * 2;
    constexpr int out_hidden_dim = 4;
    constexpr int move_dim = 6;
    constexpr int shoot_dim = 4;
    constexpr int action_dim = 5;
    // CalculateLogProbs reads both row and column indices from the action matrix,
    // so four samples exercise the same access pattern used by AIAgent.
    constexpr int batch_size = 4;

    MiniBrain::Network<MiniBrain::AutoDiffVar> preprocess_network;
    MiniBrain::Network<MiniBrain::AutoDiffVar> move_network;
    MiniBrain::Network<MiniBrain::AutoDiffVar> shoot_network;
    MiniBrain::Network<MiniBrain::AutoDiffVar> critic_network;
    MiniBrain::Random parameter_random;

    add_initialized_layer<MiniBrain::EmbeddingLayer<MiniBrain::AutoDiffVar>>(
        preprocess_network, parameter_random, input_dim, entity_feature_dim, embedding_dim);
    preprocess_network.AddLayer(std::make_unique<MiniBrain::ReLU<MiniBrain::AutoDiffVar>>());
    add_initialized_layer<MiniBrain::Attention<MiniBrain::AutoDiffVar>>(
        preprocess_network,
        parameter_random,
        entity_count * embedding_dim,
        entity_count * embedding_dim,
        embedding_dim,
        attention_key_dim);
    add_initialized_layer<MiniBrain::StatePooling<MiniBrain::AutoDiffVar>>(
        preprocess_network, parameter_random, entity_count * embedding_dim, embedding_dim);
    auto *gru = add_initialized_layer<MiniBrain::GRU<MiniBrain::AutoDiffVar>>(
        preprocess_network, parameter_random, embedding_dim * 2, gru_hidden_dim);

    add_initialized_layer<MiniBrain::FullyConnected<MiniBrain::AutoDiffVar>>(
        move_network, parameter_random, gru_hidden_dim, out_hidden_dim);
    move_network.AddLayer(std::make_unique<MiniBrain::ReLU<MiniBrain::AutoDiffVar>>());
    add_initialized_layer<MiniBrain::FullyConnected<MiniBrain::AutoDiffVar>>(
        move_network, parameter_random, out_hidden_dim, move_dim);

    add_initialized_layer<MiniBrain::FullyConnected<MiniBrain::AutoDiffVar>>(
        shoot_network, parameter_random, gru_hidden_dim, out_hidden_dim);
    shoot_network.AddLayer(std::make_unique<MiniBrain::ReLU<MiniBrain::AutoDiffVar>>());
    add_initialized_layer<MiniBrain::FullyConnected<MiniBrain::AutoDiffVar>>(
        shoot_network, parameter_random, out_hidden_dim, shoot_dim);

    add_initialized_layer<MiniBrain::FullyConnected<MiniBrain::AutoDiffVar>>(
        critic_network, parameter_random, embedding_dim * 2, embedding_dim * 4);
    critic_network.AddLayer(std::make_unique<MiniBrain::ReLU<MiniBrain::AutoDiffVar>>());
    add_initialized_layer<MiniBrain::FullyConnected<MiniBrain::AutoDiffVar>>(
        critic_network, parameter_random, embedding_dim * 4, 1);
    critic_network.SetLossFunc(std::make_unique<MiniBrain::RegressionMSE>());

    std::mt19937 input_random(20260727u);
    std::uniform_real_distribution<MiniBrain::Scalar> input_distribution(-1.0f, 1.0f);
    MiniBrain::Matrix<MiniBrain::AutoDiffVar> input(input_dim, batch_size);
    for (int i = 0; i < input.size(); ++i) {
        input(i) = input_distribution(input_random);
    }

    MiniBrain::Matrix<MiniBrain::Scalar> actions(action_dim, batch_size);
    actions <<
        0.0f, 1.0f, 2.0f, 0.0f,
        1.0f, 2.0f, 0.0f, 1.0f,
        0.8f, -0.4f, 0.3f, -0.9f,
        0.6f, 0.9f, -0.7f, 0.2f,
        1.0f, 0.0f, 1.0f, 0.0f;

    MiniBrain::Matrix<MiniBrain::Scalar> old_log_probs =
        MiniBrain::Matrix<MiniBrain::Scalar>::Zero(1, batch_size);
    MiniBrain::Matrix<MiniBrain::Scalar> advantage(1, batch_size);
    advantage << 0.75f, -0.25f, 0.5f, 1.0f;
    MiniBrain::Matrix<MiniBrain::AutoDiffVar> critic_target(1, batch_size);
    critic_target << 0.4f, -0.1f, 0.7f, 0.2f;

    MiniBrain::Adam optimizer(0.001f);
    constexpr MiniBrain::Scalar clip_epsilon = 0.2f;
    const auto critic_parameters_before = critic_network.GetParameters();

    for (int epoch = 0; epoch < 2; ++epoch) {
        gru->ResetAllMemory();

        const auto processed_data = preprocess_network.Forward(input);
        const auto move_data = move_network.Forward(processed_data);
        const auto shoot_data = shoot_network.Forward(processed_data);
        const auto critic_values = critic_network.Forward(processed_data);

        suite.expect_equal(processed_data.rows(), gru_hidden_dim, "preprocess output rows");
        suite.expect_equal(processed_data.cols(), batch_size, "preprocess output batch size");
        suite.expect_equal(move_data.rows(), move_dim, "move network output rows");
        suite.expect_equal(shoot_data.rows(), shoot_dim, "shoot network output rows");
        suite.expect_equal(critic_values.rows(), 1, "critic network output rows");

        MiniBrain::Matrix<MiniBrain::AutoDiffVar> log_probs;
        calculate_log_probs_like_aiagent(actions, move_data, shoot_data, log_probs);

        const auto ratio = (log_probs - old_log_probs.cast<MiniBrain::AutoDiffVar>()).array().exp();
        const auto surrogate1 = ratio.array() * advantage.cast<MiniBrain::AutoDiffVar>().array();
        const auto clamped_ratio = ratio.unaryExpr([](const MiniBrain::AutoDiffVar &value) {
            const MiniBrain::Scalar lower = 1.0f - clip_epsilon;
            const MiniBrain::Scalar upper = 1.0f + clip_epsilon;
            if (value.expr->val < lower) {
                return MiniBrain::AutoDiffVar(lower);
            }
            if (value.expr->val > upper) {
                return MiniBrain::AutoDiffVar(upper);
            }
            return value;
        });
        const auto surrogate2 = clamped_ratio.array() * advantage.cast<MiniBrain::AutoDiffVar>().array();
        const auto clipped_surrogate = surrogate1.matrix().cwiseMin(surrogate2.matrix());
        MiniBrain::AutoDiffVar actor_loss =
            -clipped_surrogate.sum() / static_cast<MiniBrain::Scalar>(batch_size);
        MiniBrain::RegressionMSE critic_loss_function;
        MiniBrain::AutoDiffVar critic_loss = critic_loss_function.Evaluate(critic_values, critic_target);

        suite.expect_true(std::isfinite(actor_loss.expr->val), "actor loss is finite in epoch " + std::to_string(epoch));
        suite.expect_true(std::isfinite(critic_loss.expr->val), "critic loss is finite in epoch " + std::to_string(epoch));

        preprocess_network.Backward(actor_loss);
        move_network.Backward(actor_loss);
        shoot_network.Backward(actor_loss);
        critic_network.Backward(critic_values, critic_target);

        preprocess_network.Update(optimizer);
        move_network.Update(optimizer);
        shoot_network.Update(optimizer);
        critic_network.Update(optimizer);
    }

    suite.expect_true(
        parameters_changed(critic_parameters_before, critic_network.GetParameters()),
        "critic parameters change after two backward/update epochs");
}

} // namespace

int main() {
    TestSuite suite;
    test_scalar_pooling_uses_each_batch_column_independently(suite);
    test_autodiff_pooling_preserves_values_and_expressions(suite);
    test_embedding_backward_aggregates_shared_entity_gradients(suite);
    test_aiagent_network_runs_two_training_epochs(suite);
    return suite.finish();
}
