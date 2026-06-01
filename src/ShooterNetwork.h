#pragma once

#include "MiniBrain.h"

using namespace MiniBrain;

namespace godot {
    template<typename T>
    class ShooterNetwork
    {
    protected:
        std::vector<IComputeNode<T> *> m_stateLayers;
        std::vector<IComputeNode<T> *> m_envLayers;
        std::vector<IComputeNode<T> *> m_combinedLayers;
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

        Matrix<T> Forward(const Matrix<T>& InStateData, const Matrix<T>& InEnvData);

        void Backward(T& Loss) override;

        void Update(Optimizer& opt);

        void SetLossFunc(LossFunc *lossFunc)
        {
            m_lossFunc = lossFunc;
        }
    };    
    
}