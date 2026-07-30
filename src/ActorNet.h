#pragma once

#include <MNN/expr/Module.hpp>

namespace MiniMind
{
    
    class ActorNet : public MNN::Express::Module
    {
    public:
        ActorNet(int input_size, int move_output_size, int shoot_output_size);
        virtual ~ActorNet() override = default;

        std::vector<MNN::Express::VARP> onForward(
        const std::vector<MNN::Express::VARP>& inputs) override;

        MNN::Express::Module* clone(CloneContext* context) const override;

        void ResetAllGRUMemory();
        void ResetGRUMemory(int batch_index);

    protected:
        ActorNet() = default;
        int m_input_size = 0;
        int m_move_output_size = 0;
        int m_shoot_output_size = 0;

        MNN::Express::VARP m_preprocess_embedding;
        MNN::Express::VARP m_preprocess_attention;
        MNN::Express::VARP m_preprocess_state_pooling;
        MNN::Express::VARP m_preprocess_gru;

        MNN::Express::VARP m_move_fc;
        MNN::Express::VARP m_move_fc_out;

        MNN::Express::VARP m_shoot_fc;
        MNN::Express::VARP m_shoot_fc_out;

        MNN::Express::VARP m_gru_hidden;
    };
}