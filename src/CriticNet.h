#pragma once

#include <MNN/expr/Module.hpp>

namespace MiniMind {
    class CriticNet : public MNN::Express::Module {
    public:
        CriticNet(int input_size, int hidden_size);
        virtual ~CriticNet() override = default;

        std::vector<MNN::Express::VARP> onForward(
            const std::vector<MNN::Express::VARP>& inputs) override;

        MNN::Express::Module* clone(CloneContext* context) const override;

    protected:
        CriticNet() = default;
        int m_input_size = 0;
        int m_hidden_size = 0;

        MNN::Express::VARP m_fc;
        MNN::Express::VARP m_fc_bias;
        MNN::Express::VARP m_fc_out;
        MNN::Express::VARP m_fc_out_bias;
    };
} // namespace MiniMind
