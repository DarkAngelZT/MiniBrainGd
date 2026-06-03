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

void AIAgent::Init() 
{
    // Placeholder initialization. Real network setup can be done later.
    UtilityFunctions::print("AIAgent: Init called");
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