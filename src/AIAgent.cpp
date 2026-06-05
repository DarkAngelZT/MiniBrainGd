#include "AIAgent.h"
#include <godot_cpp/core/class_db.hpp>
#include <godot_cpp/core/utility_functions.hpp>

using namespace godot;

void AIAgent::_bind_methods() 
{
    ClassDB::bind_method(D_METHOD("Init"), &AIAgent::Init);
    ClassDB::bind_method(D_METHOD("get_mode"), &AIAgent::get_mode);
    ClassDB::bind_method(D_METHOD("ProcessSensorData", "data"), &AIAgent::ProcessSensorData);
    ClassDB::bind_method(D_METHOD("PushTrainingData", "batch_inputs", "batch_targets"), &AIAgent::PushTrainingData);
    ClassDB::bind_method(D_METHOD("Train", "step"), &AIAgent::Train);

    BIND_ENUM_CONSTANT(TRAINING);
    BIND_ENUM_CONSTANT(INFERENCE);
}

AIAgent::AIAgent(AIAgentMode mode) : mode(mode) 
{
    if (mode == AIAgentMode::Inference) {
        network = new MiniBrain::Network<MiniBrain::Scalar>();
    } else {
        actor = new MiniBrain::Network<MiniBrain::AutoDiffVar>();
        critic = new MiniBrain::Network<MiniBrain::AutoDiffVar>();
        m_training_input = new MiniBrain::MatrixX<MiniBrain::AutoDiffVar>();
        m_training_target = new MiniBrain::MatrixX<MiniBrain::AutoDiffVar>();
    }
}

AIAgent::~AIAgent() 
{
    if (mode == AIAgentMode::Inference) 
    {
        delete network;
    } 
    else
    {
        delete actor;
        delete critic;
        delete m_training_input;
        delete m_training_target;
    }
}

AIAgentMode AIAgent::get_mode() const {
    return mode;
}

void AIAgent::Init(
    int input_dim, int output_dim,
     int entity_feature_dim, int embedding_dim, 
     int attention_key_dim, int gru_hidden_dim, 
     int out_hidden_dim) 
{
   if (mode == AIAgentMode::Inference) {
        // 构建推理网络
        m_preprocessNet = new MiniBrain::Network<MiniBrain::Scalar>();
        m_moveNet = new MiniBrain::Network<MiniBrain::Scalar>();
        m_shootNet = new MiniBrain::Network<MiniBrain::Scalar>();

        m_preprocessNet->AddLayer(std::make_unique<MiniBrain::FullyConnectedLayer<MiniBrain::Scalar>>(input_dim, embedding_dim));
        m_preprocessNet->AddLayer(std::make_unique<MiniBrain::ReLULayer<MiniBrain::Scalar>>());
        m_preprocessNet->AddLayer(std::make_unique<MiniBrain::AttentionLayer<MiniBrain::Scalar>>(embedding_dim, attention_key_dim));
        m_preprocessNet->AddLayer(std::make_unique<MiniBrain::StatePoolingLayer<MiniBrain::Scalar>>(embedding_dim, entity_feature_dim));
        m_preprocessNet->AddLayer(std::make_unique<MiniBrain::GRULayer<MiniBrain::Scalar>>(embedding_dim, gru_hidden_dim));

        m_moveNet->AddLayer(std::make_unique<MiniBrain::FullyConnectedLayer<MiniBrain::Scalar>>(gru_hidden_dim, out_hidden_dim));
        m_moveNet->AddLayer(std::make_unique<MiniBrain::ReLULayer<MiniBrain::Scalar>>());
        m_moveNet->AddLayer(std::make_unique<MiniBrain::FullyConnectedLayer<MiniBrain::Scalar>>(out_hidden_dim, output_dim));

        m_shootNet->AddLayer(std::make_unique<MiniBrain::FullyConnectedLayer<MiniBrain::Scalar>>(gru_hidden_dim, out_hidden_dim));
        m_shootNet->AddLayer(std::make_unique<MiniBrain::ReLULayer<MiniBrain::Scalar>>());
        m_shootNet->AddLayer(std::make_unique<MiniBrain::FullyConnectedLayer<MiniBrain::Scalar>>(out_hidden_dim, output_dim));
    } else {
        // 构建训练网络
        m_actor_preprocessNet = new MiniBrain::Network<MiniBrain::AutoDiffVar>();
        m_actor_moveNet = new MiniBrain::Network<MiniBrain::AutoDiffVar>();
        m_actor_shootNet = new MiniBrain::Network<MiniBrain::AutoDiffVar>();

        m_criticNet = new MiniBrain::Network<MiniBrain::AutoDiffVar>();

        // 这里可以添加具体的网络层构建逻辑，例如全连接层、注意力机制等
     }   
}

PackedFloat32Array AIAgent::ProcessSensorData(const PackedFloat32Array &input) 
{
    godot::PackedFloat32Array output_array;

    if (!network || input.size() == 0) {
        if (!network) godot::UtilityFunctions::push_error("Agent: Network is not set!");
        return output_array;
    }

    const int input_size = input.size();

    MiniBrain::Matrix input_matrix(input_size, 1);
    
    std::copy(input.ptr(), input.ptr() + input_size, input_matrix.data());

    MiniBrain::Matrix output_matrix = network->Forward(input_matrix);

    const int output_size = output_matrix.rows();
    output_array.resize(output_size);

    std::copy(output_matrix.data(), output_matrix.data() + output_size, 
              output_array.ptrw());

    return output_array;
}

void AIAgent::PushTrainingData(const godot::Array& batch_inputs, const godot::Array& batch_targets) 
{
    if (!network) {
        godot::UtilityFunctions::push_error("Agent: Network is not set!");
        return;
    }

    if (batch_inputs.size() == 0 || batch_targets.size() == 0) {
        godot::UtilityFunctions::push_error("Agent: Batch input or target array is empty!");
        return;
    }

    const int batch_size = batch_inputs.size();
    MatrixXv input_matrix(input_dim, batch_size);

    for (int i = 0; i < batch_size; ++i) {
        godot::PackedFloat32Array sample = batch_inputs[i];
        
        if (sample.size() != input_dim) {
            godot::UtilityFunctions::push_error("Agent: Input sample " + godot::String::num(i) + " size mismatch!");
            return;
        }        

        godot::PackedFloat32Array target = batch_targets[i];
        
        if (target.size() != output_dim) {
            godot::UtilityFunctions::push_error("Agent: Target sample " + godot::String::num(i) + " size mismatch!");
            return;
        }

        std::copy(sample.ptr(), sample.ptr() + input_dim, 
                        input_matrix.col(i).data());

        // 使用 std::copy
        std::copy(target.ptr(), target.ptr() + output_dim, 
                  target_matrix.col(i).data());
    }
}

void AIAgent::Train(int step) 
{
    UtilityFunctions::print("AIAgent: Train called");
    for (int i = 0; i < step; ++i) {
        // Placeholder training loop. Real training logic can be implemented later.
        UtilityFunctions::print("Training step: " + godot::String::num(i + 1));
    }
}