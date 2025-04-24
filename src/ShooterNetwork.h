#pragma once

#include "..\MiniBrain\Source\MiniBrain.h"

using namespace MiniBrain;

namespace godot {
    class ShooterNetwork
    {
    protected:
        FullyConnected* m_stateFCLayer;
        std::vector<IComputeNode *> m_layers;
        LossFunc *m_lossFunc=nullptr;
    public:
        ShooterNetwork(/* args */);
        ~ShooterNetwork();
    };    
    
}