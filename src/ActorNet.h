#pragma once

#include <MNN/expr/Module.hpp>
#include <MNN/expr/NeuralNetWorkOp.hpp>

#include <memory>

namespace MiniMind {

class Attention;
class Embedding;
class GRU;
class StatePooling;

class ActorNet : public MNN::Express::Module {
public:
    // 便捷构造沿用AIAgent的默认网络宽度；input_size表示单个实体的特征数。
    ActorNet(int input_size, int move_output_size, int shoot_output_size);

    ActorNet(int entity_feature_dim,
             int move_output_size,
             int shoot_output_size,
             int embedding_dim,
             int attention_key_dim,
             int gru_hidden_dim,
             int out_hidden_dim);

    virtual ~ActorNet() override = default;

    // 输入为state[B,E,F]，输出依次为move、shoot和当前步hidden；GRU记忆由模块内部维护。
    std::vector<MNN::Express::VARP> onForward(
        const std::vector<MNN::Express::VARP>& inputs) override;

    MNN::Express::Module* clone(CloneContext* context) const override;

    void ResetAllGRUMemory();
    void ResetGRUMemory(int batch_index);
    void CacheMemory();
    void RestoreMemory();
    MNN::Express::VARP GetGRUMemory() const { return MNN::Express::_Clone(m_gru_hidden, true); }
    void SetGRUMemory(const MNN::Express::VARP& memory) { m_gru_hidden = MNN::Express::_Clone(memory, true); }

protected:
    ActorNet() = default;

    int m_input_size = 0;
    int m_move_output_size = 0;
    int m_shoot_output_size = 0;
    int m_embedding_dim = 0;
    int m_attention_key_dim = 0;
    int m_gru_hidden_dim = 0;
    int m_out_hidden_dim = 0;

    std::shared_ptr<Embedding> m_preprocess_embedding;
    std::shared_ptr<Attention> m_preprocess_attention;
    std::shared_ptr<StatePooling> m_preprocess_state_pooling;
    std::shared_ptr<GRU> m_preprocess_gru;

    MNN::Express::VARP m_move_fc;
    MNN::Express::VARP m_move_fc_bias;
    MNN::Express::VARP m_move_fc_out;
    MNN::Express::VARP m_move_fc_out_bias;

    MNN::Express::VARP m_shoot_fc;
    MNN::Express::VARP m_shoot_fc_bias;
    MNN::Express::VARP m_shoot_fc_out;
    MNN::Express::VARP m_shoot_fc_out_bias;

    MNN::Express::VARP m_gru_hidden;
    MNN::Express::VARP m_gru_hidden_cache;
};

} // MiniMind命名空间
