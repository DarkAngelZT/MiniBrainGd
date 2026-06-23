#include "AIAgent.h"
#include <godot_cpp/core/class_db.hpp>
#include <godot_cpp/core/utility_functions.hpp>

using namespace godot;

void AIAgent::_bind_methods() 
{
    ClassDB::bind_method(D_METHOD("Init", "input_dim", "output_dim", "entity_feature_dim", "embedding_dim", "attention_key_dim", "gru_hidden_dim", "out_hidden_dim"), &AIAgent::Init, DEFVAL(16), DEFVAL(16), DEFVAL(128), DEFVAL(128));
    ClassDB::bind_method(D_METHOD("get_mode"), &AIAgent::get_mode);
    ClassDB::bind_method(D_METHOD("ProcessSensorData", "data"), &AIAgent::ProcessSensorData);
    ClassDB::bind_method(D_METHOD("BatchProcessSensorData", "batch_data", "agent_ids"), &AIAgent::BatchProcessSensorData);
    ClassDB::bind_method(D_METHOD("PushTrainingData", "batch_rewards", "agent_ids", "batch_dones"), &AIAgent::PushTrainingData);
    ClassDB::bind_method(D_METHOD("Train", "step"), &AIAgent::Train);
    ClassDB::bind_method(D_METHOD("SetBatchSize", "batch_size"), &AIAgent::SetBatchSize);

    BIND_ENUM_CONSTANT(TRAINING);
    BIND_ENUM_CONSTANT(INFERENCE);
}

void godot::AIAgent::CalculateLogProbs(
    const MiniBrain::Matrix<MiniBrain::AutoDiffVar> &action_new,
     const MiniBrain::Matrix<MiniBrain::AutoDiffVar> &moveData,
     const MiniBrain::Matrix<MiniBrain::AutoDiffVar> &shootData,
     MiniBrain::Matrix<MiniBrain::AutoDiffVar> &log_probs)
{
    int batch_size = action_new.cols();
    log_probs.resize(1, batch_size);
    for (int i = 0; i < batch_size; i++) {
        MiniBrain::AutoDiffVar log_p_horizon = 0.0;
        MiniBrain::AutoDiffVar log_p_vertical = 0.0;
        MiniBrain::AutoDiffVar log_p_angle = 0.0;

        MiniBrain::Scalar act_horiz = action_new(i, 0).expr->val;
        MiniBrain::Scalar act_vert  = action_new(i, 1).expr->val;
        MiniBrain::Scalar act_angle = action_new(i, 2).expr->val;

        // --- 1. 计算水平移动的 Log Prob (Categorical Softmax LogProb) ---
        if (move_rows >= 3) {
            MiniBrain::AutoDiffVar v0 = moveData(0, col);
            MiniBrain::AutoDiffVar v1 = moveData(1, col);
            MiniBrain::AutoDiffVar v2 = moveData(2, col);

            // 稳定的 Log-Sum-Exp 或者是标准的 Softmax Log 形式
            MiniBrain::AutoDiffVar max_v = std::max({v0.expr->val, v1.expr->val, v2.expr->val}); // 提升数值稳定性
            MiniBrain::AutoDiffVar log_z = max_v + std::log(std::exp(v0 - max_v) + std::exp(v1 - max_v) + std::exp(v2 - max_v));
            
            // 根据实际选中的动作提取对应的对数概率： log(matrix(act, col)) - log_z
            if (act_horiz == 0) log_p_horizon = v0 - log_z;
            else if (act_horiz == 1) log_p_horizon = v1 - log_z;
            else log_p_horizon = v2 - log_z;
        }

        // --- 2. 计算垂直移动的 Log Prob ---
        if (move_rows >= 6) {
            MiniBrain::AutoDiffVar v3 = moveData(3, col);
            MiniBrain::AutoDiffVar v4 = moveData(4, col);
            MiniBrain::AutoDiffVar v5 = moveData(5, col);
            
            MiniBrain::AutoDiffVar max_v = std::max({v3.expr->val, v4.expr->val, v5.expr->val});
            MiniBrain::AutoDiffVar log_z = max_v + std::log(std::exp(v3 - max_v) + std::exp(v4 - max_v) + std::exp(v5 - max_v));
            
            if (act_vert == 0) log_p_vertical = v3 - log_z;
            else if (act_vert == 1) log_p_vertical = v4 - log_z;
            else log_p_vertical = v5 - log_z;
        }

        // --- 3. 计算高斯分布射击角度的 Log Prob (Gaussian LogProb) ---
        MiniBrain::AutoDiffVar mean = shootData(0, col);
        MiniBrain::AutoDiffVar var  = shootData(1, col);
        // 确保方差严格大于 0（可以使用 softplus 或者 abs + epsilon）
        MiniBrain::AutoDiffVar std_dev_sq = std::abs(var) + 1e-6; 

        // 高斯分布对数概率公式： -0.5 * ln(2 * pi * sigma^2) - (x - mu)^2 / (2 * sigma^2)
        const double PI = 3.141592653589793;
        log_p_angle = -0.5 * std::log(2.0 * PI * std_dev_sq) - std::pow(act_angle - mean, 2) / (2.0 * std_dev_sq);

        // --- 4. 对应 PyTorch 中的 .sum(1) 操作 ---
        // 独立动作的对数概率直接相加
        log_probs(0, col) = log_p_horizon + log_p_vertical + log_p_angle;
    }
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

    MiniBrain::Scalar horizon = 0;
    MiniBrain::Scalar vertical = 0;
    const int move_rows = moveData.rows();
    if (move_rows >= 3) {
        int max_index = 0;
        MiniBrain::Scalar max_value = moveData(0, 0);
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
        MiniBrain::Scalar max_value = moveData(max_index, 0);
        for (int r = move_rows - 2; r < move_rows; ++r) {
            if (moveData(r, 0) > max_value) {
                max_value = moveData(r, 0);
                max_index = r;
            }
        }
        vertical = max_index - 3;
    }

    MiniBrain::Scalar shootAngle = std::tanh(shootData(0, 0))*180;
    MiniBrain::Scalar shootProb = 1.0f / (1.0f + exp(-shootData(2, 0)));
    MiniBrain::Scalar shootAction = shootProb > 0.5 ? 1.0 : 0.0;

    const int output_size = 4;
    output_array.resize(output_size);
    MiniBrain::Matrix<Scalar> combinedOutput(output_size, 1);
    combinedOutput << horizon, vertical, shootAngle, shootAction;

    std::copy(combinedOutput.data(), combinedOutput.data() + output_size, output_array.ptrw());

    return output_array;
}

godot::Array godot::AIAgent::BatchProcessSensorData(const godot::Array &batch_data, const godot::PackedInt32Array &agent_ids)
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
    m_training_data->buffer_input.setZero();
    m_training_data>buffer_action.setZero();
    m_training_data->input_mapping.clear();
    for (int i = 0; i < batch_size; ++i) {
        godot::PackedFloat32Array sample = batch_inputs[i];
        
        if (sample.size() != m_insize) {
            godot::UtilityFunctions::push_error("Agent: Input sample " + godot::String::num(i) + " size mismatch!");
            return;
        }        

        for (int in = 0; in < m_insize; in++)
        {
            input_matrix(in, i) = sample[in];
            m_training_data->buffer_input(in, i) = sample[in];
        }
        m_training_data->input_mapping[agent_ids[i]] = i;
    }
    auto processedData = m_actor_preprocessNet->Forward(input_matrix);
    auto moveData = m_actor_moveNet->Forward(processedData);
    auto shootData = m_actor_shootNet->Forward(processedData);

    static std::mt19937 gen(std::random_device{}());

    godot::Array batch_array;
    batch_array.resize(batch_size);

    const int output_size= 4;

    for (int col = 0; col < batch_size; ++col) 
    {
        MiniBrain::Scalar horizon = 0;
        MiniBrain::Scalar vertical = 0;
        godot::PackedFloat32Array output_array;
        output_array.resize(output_size);
        MiniBrain::Vector<MiniBrain::Scalar> action(output_size);

        if (move_rows >= 3) 
        {
            MiniBrain::Scalar w0 = moveData(0, col).expr->val;
            MiniBrain::Scalar w1 = moveData(1, col).expr->val;
            MiniBrain::Scalar w2 = moveData(2, col).expr->val;

            // 如果权重是网络原始输出，需通过 Softmax 或 Exp 确保为正数
            w0 = std::exp(w0); w1 = std::exp(w1); w2 = std::exp(w2); 

            std::discrete_distribution<> dist_horiz({w0, w1, w2});
            horizon = dist_horiz(gen); // 随机生成 0, 1, 2
        }

        if (move_rows >= 6) 
        {
            MiniBrain::Scalar w0 = moveData(3, col).expr->val;
            MiniBrain::Scalar w1 = moveData(4, col).expr->val;
            MiniBrain::Scalar w2 = moveData(5, col).expr->val;

            w0 = std::exp(w0); w1 = std::exp(w1); w2 = std::exp(w2);

            std::discrete_distribution<> dist_vert({w0, w1, w2});
            vertical = dist_vert(gen); // 随机生成 0, 1, 2
        }

        MiniBrain::Scalar mean = shootData(0, col).expr->val; // 直接使用网络输出作为均值
        MiniBrain::Scalar var  = shootData(1, col).expr->val;
        
        // 确保方差为正数（网络输出可能为负），取绝对值并加上微小值防止为0
        MiniBrain::Scalar std_dev = std::sqrt(std::abs(var) + 1e-6); 

        std::normal_distribution<MiniBrain::Scalar> dist_angle(mean, std_dev);
        MiniBrain::Scalar shootAngle = dist_angle(gen);

        action(2) = shootAngle;

        shootAngle = std::tanh(shootAngle)*180;

        // --- 4. 处理射击动作 (第 2 行，原概率逻辑不变) ---
        MiniBrain::Scalar shoot_logits = shootData(2, col).expr->val;
        MiniBrain::Scalar shootProb = 1.0f / (1.0f + std::exp(-shoot_logits));
        MiniBrain::Scalar shootAction = shootProb > 0.5 ? 1.0 : 0.0;

        action(0) = horizon;
        action(1) = vertical;
        action(3) = shoot_logits;

        m_training_data->buffer_action.col(col) = action;

        output_array[0] = horizon;
        output_array[1] = vertical;
        output_array[2] = shootAngle;
        output_array[3] = shootAction;

        batch_array[col] = output_array;
    }

    return batch_array;
}

void AIAgent::PushTrainingData(const godot::PackedFloat32Array& batch_rewards, const godot::PackedInt32Array &agent_ids, const godot::PackedFloat32Array &batch_dones) 
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

    for (int i = 0; i < batch_size; ++i) {
        float rewards = batch_rewards[i];  
        float done = batch_dones[i];
        int id = agent_ids[i];
        int index = m_training_data->agent_write_index[id];
        int buffer_index = m_training_data->input_mapping[id];
        m_training_data->agent_write_index[id] += batch_size; // Increment the write index for this agent

        m_training_data->state.col(index) = m_training_data->buffer_input.col(buffer_index);
        m_training_data->actions.col(index) = m_training_data->buffer_action.col(buffer_index);
        m_training_data->rewards(0, index) = rewards;
        m_training_data->done(0, index) = done;
    }
}

void AIAgent::Train(int step) 
{
    for (int epoch = 0; epoch < step; ++epoch) 
    {
        for (int f = 0; f < m_training_data->num_frames; f++)
        {
            //顺序跑每一帧
            MiniBrain::Matrix<MiniBrain::Scalar> batch_states = m_training_data->state.block(
                0, f * m_training_data->batch_size, m_training_data->state.rows(), m_training_data->batch_size);
            MiniBrain::Matrix<MiniBrain::Scalar> batch_actions = m_training_data->actions.block(
                0, f * m_training_data->batch_size, m_training_data->actions.rows(),m_training_data->batch_size);
            MiniBrain::Matrix<MiniBrain::Scalar> batch_rewards = m_training_data->rewards.block(
                0, f * m_training_data->batch_size, m_training_data->rewards.rows(), m_training_data->batch_size);
            MiniBrain::Matrix<MiniBrain::Scalar> batch_dones = m_training_data->done.block(
                0, f * m_training_data->batch_size, m_training_data->done.rows(), m_training_data->batch_size);

            MiniBrain::Matrix<MiniBrain::AutoDiffVar> input_matrix = batch_states.cast<MiniBrain::AutoDiffVar>();
            MiniBrain::Matrix<MiniBrain::AutoDiffVar> action_new = batch_actions.cast<MiniBrain::AutoDiffVar>();
        }
        //计算PPO
        //更新
    }
}

void godot::AIAgent::SetBatchInfo(int batch_size, int action_dim, int num_frames)
{
    if(mode == AIAgentMode::Training)
    {
        m_actor_GRLayer->SetBatchSize(batch_size);
        m_training_data->Init(batch_size, num_frames, m_insize, action_dim);
    }
    else
    {
        m_GRULayer->SetBatchSize(batch_size);
    }

}
