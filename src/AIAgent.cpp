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
}

AIAgent::~AIAgent() 
{
    if (mode == AIAgentMode::Inference) 
    {
        delete m_preprocessNet;
        delete m_moveNet;
        delete m_shootNet;
        m_GRULayer = nullptr;
    } 
    else
    {
        delete m_actor_preprocessNet;
        delete m_actor_moveNet;
        delete m_actor_shootNet;
        m_actor_GRULayer = nullptr;

        delete m_criticNet;

        m_training_data.reset();
    }
}

AIAgentMode AIAgent::get_mode() const {
    return mode;
}

void AIAgent::Init(
    int input_dim, int move_dim, int shoot_dim,
     int entity_feature_dim, int embedding_dim, 
     int attention_key_dim, int gru_hidden_dim, 
     int out_hidden_dim) 
{
    m_insize = input_dim;
    m_outSize = move_dim + shoot_dim;
    int n_entities = input_dim / entity_feature_dim;
    if (mode == AIAgentMode::Inference) 
    {
        // 构建推理网络
        m_preprocessNet = new MiniBrain::Network<MiniBrain::Scalar>();
        m_moveNet = new MiniBrain::Network<MiniBrain::Scalar>();
        m_shootNet = new MiniBrain::Network<MiniBrain::Scalar>();

        m_preprocessNet->AddLayer(std::make_unique<MiniBrain::EmbeddingLayer<MiniBrain::Scalar>>(input_dim, entity_feature_dim, embedding_dim));
        m_preprocessNet->AddLayer(std::make_unique<MiniBrain::ReLULayer<MiniBrain::Scalar>>());
        m_preprocessNet->AddLayer(std::make_unique<MiniBrain::AttentionLayer<MiniBrain::Scalar>>(n_entities * embedding_dim, n_entities * embedding_dim, embedding_dim, attention_key_dim));
        m_preprocessNet->AddLayer(std::make_unique<MiniBrain::StatePoolingLayer<MiniBrain::Scalar>>(n_entities * embedding_dim, embedding_dim));
        auto gru_layer = std::make_unique<MiniBrain::GRULayer<MiniBrain::Scalar>>(embedding_dim, gru_hidden_dim);
        m_GRULayer = gru_layer.get();
        m_preprocessNet->AddLayer(gru_layer);

        m_moveNet->AddLayer(std::make_unique<MiniBrain::FullyConnectedLayer<MiniBrain::Scalar>>(gru_hidden_dim, out_hidden_dim));
        m_moveNet->AddLayer(std::make_unique<MiniBrain::ReLULayer<MiniBrain::Scalar>>());
        m_moveNet->AddLayer(std::make_unique<MiniBrain::FullyConnectedLayer<MiniBrain::Scalar>>(out_hidden_dim, move_dim));

        m_shootNet->AddLayer(std::make_unique<MiniBrain::FullyConnectedLayer<MiniBrain::Scalar>>(gru_hidden_dim, out_hidden_dim));
        m_shootNet->AddLayer(std::make_unique<MiniBrain::ReLULayer<MiniBrain::Scalar>>());
        m_shootNet->AddLayer(std::make_unique<MiniBrain::FullyConnectedLayer<MiniBrain::Scalar>>(out_hidden_dim, shoot_dim));
    }
    else 
    {
        // 构建训练网络
        m_actor_preprocessNet = new MiniBrain::Network<MiniBrain::AutoDiffVar>();
        m_actor_moveNet = new MiniBrain::Network<MiniBrain::AutoDiffVar>();
        m_actor_shootNet = new MiniBrain::Network<MiniBrain::AutoDiffVar>();

        m_criticNet = new MiniBrain::Network<MiniBrain::AutoDiffVar>();

        m_training_data = std::make_shared<TrainingData>();

        m_actor_preprocessNet->AddLayer(std::make_unique<MiniBrain::EmbeddingLayer<MiniBrain::AutoDiffVar>>(input_dim, entity_feature_dim, embedding_dim));
        m_actor_preprocessNet->AddLayer(std::make_unique<MiniBrain::ReLULayer<MiniBrain::AutoDiffVar>>());
        m_actor_preprocessNet->AddLayer(std::make_unique<MiniBrain::AttentionLayer<MiniBrain::AutoDiffVar>>(n_entities * embedding_dim, n_entities * embedding_dim, embedding_dim, attention_key_dim));
        m_actor_preprocessNet->AddLayer(std::make_unique<MiniBrain::StatePoolingLayer<MiniBrain::AutoDiffVar>>(n_entities * embedding_dim, embedding_dim));
        auto gru_layer_train = std::make_unique<MiniBrain::GRULayer<MiniBrain::AutoDiffVar>>(embedding_dim, gru_hidden_dim);
        m_actor_GRULayer = gru_layer_train.get();
        m_actor_preprocessNet->AddLayer(gru_layer_train);

        m_actor_moveNet->AddLayer(std::make_unique<MiniBrain::FullyConnectedLayer<MiniBrain::AutoDiffVar>>(gru_hidden_dim, out_hidden_dim));
        m_actor_moveNet->AddLayer(std::make_unique<MiniBrain::ReLULayer<MiniBrain::AutoDiffVar>>());
        m_actor_moveNet->AddLayer(std::make_unique<MiniBrain::FullyConnectedLayer<MiniBrain::AutoDiffVar>>(out_hidden_dim, move_dim));

        m_actor_shootNet->AddLayer(std::make_unique<MiniBrain::FullyConnectedLayer<MiniBrain::AutoDiffVar>>(gru_hidden_dim, out_hidden_dim));
        m_actor_shootNet->AddLayer(std::make_unique<MiniBrain::ReLULayer<MiniBrain::AutoDiffVar>>());
        m_actor_shootNet->AddLayer(std::make_unique<MiniBrain::FullyConnectedLayer<MiniBrain::AutoDiffVar>>(out_hidden_dim, shoot_dim));
    }   
}

PackedFloat32Array AIAgent::ProcessSensorData(const PackedFloat32Array &input) 
{
    godot::PackedFloat32Array output_array;

    if (!m_preprocessNet || !m_moveNet || !m_shootNet || input.size() == 0) {
        godot::UtilityFunctions::push_error("Agent: Network is not set!");
        return output_array;
    }

    const int input_size = input.size();

    MiniBrain::Matrix input_matrix(input_size, 1);
    
    std::copy(input.ptr(), input.ptr() + input_size, input_matrix.data());

    MiniBrain::Matrix<Scalar> processedData = m_preprocessNet->Forward(input_matrix);

    MiniBrain::Matrix<Scalar> moveData = m_moveNet->Forward(processedData);
    MiniBrain::Matrix<Scalar> shootData = m_shootNet->Forward(processedData);

    Scalar horizon = 0;
    Scalar vertical = 0;
    const int move_rows = moveData.rows();
    if (move_rows >= 3) {
        int max_index = 0;
        Scalar max_value = moveData(0, 0);
        for (int r = 1; r < 3; ++r) {
            if (moveData(r, 0) > max_value) {
                max_value = moveData(r, 0);
                max_index = r;
            }
        }
        horizon = max_index;
    }
    if (move_rows >= 6) {
        int max_index = move_rows - 3;
        Scalar max_value = moveData(max_index, 0);
        for (int r = move_rows - 2; r < move_rows; ++r) {
            if (moveData(r, 0) > max_value) {
                max_value = moveData(r, 0);
                max_index = r;
            }
        }
        vertical = max_index - 3;
    }

    Scalar shootAngle = std::tanh(shootData(0, 0))*180;
    Scalar shootProb = 1.0f / (1.0f + exp(-shootData(2, 0)));
    Scalar shootAction = shootProb > 0.5 ? 1.0 : 0.0;

    const int output_size = 4;
    output_array.resize(output_size);
    MiniBrain::Matrix<Scalar> combinedOutput(output_size, 1);
    combinedOutput << horizon, vertical, shootAngle, shootAction;

    std::copy(combinedOutput.data(), combinedOutput.data() + output_size, output_array.ptrw());

    return output_array;
}

godot::Array godot::AIAgent::BatchProcessSensorData(const godot::Array &batch_data, const godot::Array &agent_ids))
{
    if (!m_actor_preprocessNet || !m_actor_moveNet || !m_actor_shootNet) {
        godot::UtilityFunctions::push_error("Agent: Network is not set!");
        return;
    }

    if (batch_data.size() == 0 ) {
        godot::UtilityFunctions::push_error("Agent: Batch input or target array is empty!");
        return;
    }

    const int batch_size = batch_inputs.size();
    MiniBrain::MatrixX<MiniBrain::AutoDiffVar> input_matrix(m_insize, batch_size);
    for (int i = 0; i < batch_size; ++i) {
        godot::PackedFloat32Array sample = batch_inputs[i];
        
        if (sample.size() != m_insize) {
            godot::UtilityFunctions::push_error("Agent: Input sample " + godot::String::num(i) + " size mismatch!");
            return;
        }        

        for (int in = 0; in < m_insize; in++)
        {
            input_matrix(in, i) = sample[in];
        }
    }
    auto processedData = m_actor_preprocessNet->Forward(input_matrix);
    auto moveData = m_actor_moveNet->Forward(processedData);
    auto shootData = m_actor_shootNet->Forward(processedData);
    
    Matrix<MiniBrain::AutoDiffVar> combinedOutput(m_outSize, batch_size);
    combinedOutput.topRows(moveData.rows()) = moveData;
    combinedOutput.bottomRows(shootData.rows()) = shootData;

    godot::Array batch_array;

    batch_array.resize(batch_size);
    for (int i = 0; i < batch_size; ++i) {
        godot::PackedFloat32Array output_array;
        output_array.resize(m_outSize);
        float* write_ptr = output_array.ptrw();
        for (int out = 0; out < m_outSize; out++)
        {
            write_ptr[out] = combinedOutput(out, i).expr.val;
        }
        batch_array[i] = output_array;
    }

    return batch_array;
}

void AIAgent::PushTrainingData(const godot::Array& batch_rewards, const godot::Array& agent_ids, const godot::Array& batch_dones) 
{
    if (!m_actor_preprocessNet || !m_actor_moveNet || !m_actor_shootNet || !m_criticNet) {
        godot::UtilityFunctions::push_error("Agent: Network is not set!");
        return;
    }

    if (batch_rewards.size() == 0 || agent_ids.size() == 0 || batch_dones.size() == 0) {
        godot::UtilityFunctions::push_error("Agent: One or more training data arrays are empty!");
        return;
    }

    const int batch_size = batch_rewards.size();
    MiniBrain::MatrixX<MiniBrain::AutoDiffVar> input_matrix(m_insize, batch_size);

    for (int i = 0; i < batch_size; ++i) {
        godot::PackedFloat32Array sample = batch_inputs[i];
        
        if (sample.size() != m_insize) {
            godot::UtilityFunctions::push_error("Agent: Input sample " + godot::String::num(i) + " size mismatch!");
            return;
        }        

        godot::PackedFloat32Array target = batch_targets[i];
        
        if (target.size() != m_outSize) {
            godot::UtilityFunctions::push_error("Agent: Target sample " + godot::String::num(i) + " size mismatch!");
            return;
        }

        std::copy(sample.ptr(), sample.ptr() + m_insize, 
                        input_matrix.col(i).data());

        std::copy(target.ptr(), target.ptr() + m_outSize, 
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