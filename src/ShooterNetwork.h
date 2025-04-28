#pragma once

#include "..\MiniBrain\Source\MiniBrain.h"

using namespace MiniBrain;

namespace godot {
    class ShooterNetwork
    {
    protected:
        std::vector<IComputeNode *> m_stateLayers;
        std::vector<IComputeNode *> m_envLayers;
        std::vector<IComputeNode *> m_combinedLayers;
        LossFunc *m_lossFunc=nullptr;
    public:
        ShooterNetwork(
            int StateDim, 
            int EnvWidth, int EnvHeight, int EnvChannel,
            int OutDim,
            int StateHiddenDim,
            int EnvHiddenWidth,int EnvHiddenHeight, int EnvHiddenChannel,
            int CombinedHiddenDim);
        ~ShooterNetwork();

        void Forward(const Matrix& InData);

        void Backward(const Matrix& Input, const Matrix& Target);

        void Update(Optimizer& opt);

        void SetLossFunc(LossFunc *lossFunc)
        {
            m_lossFunc = lossFunc;
        }
    };    
    
}