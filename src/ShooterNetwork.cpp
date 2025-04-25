#include "ShooterNetwork.h"

namespace godot {
    ShooterNetwork::ShooterNetwork(
            int StateDim, 
            int EnvWidth, int EnvHeight, int EnvChannel,
            int OutDim,
            int StateHiddenDim,
            int EnvHiddenWidth,int EnvHiddenHeight, int EnvHiddenChannel,
            int CombinedHiddenDim)
    {
        m_stateLayers.push_back(new FullyConnected(StateDim, StateHiddenDim));
        m_stateLayers.push_back(new ReLU());

        m_envLayers.push_back(new Convolutional(EnvWidth,EnvHeight,EnvChannel,EnvHiddenChannel,EnvHiddenWidth,EnvHiddenHeight));
        m_envLayers.push_back(new ReLU());

        int combinedDim = StateHiddenDim+EnvHiddenWidth*EnvHiddenHeight*EnvHiddenChannel;
        m_combinedLayers.push_back(new FullyConnected(combinedDim,CombinedHiddenDim));
        m_combinedLayers.push_back(new ReLU());
        m_combinedLayers.push_back(new GRU(CombinedHiddenDim,CombinedHiddenDim));
        m_combinedLayers.push_back(new ReLU());
        m_combinedLayers.push_back(new FullyConnected(CombinedHiddenDim, OutDim));
    }

    ShooterNetwork::~ShooterNetwork()
    {
        for (int i = 0; i < m_stateLayers.size(); i++)
        {
            delete m_stateLayers[i];
        }
        for (int i = 0; i < m_envLayers.size(); i++)
        {
            delete m_envLayers[i];
        }
        for (int i = 0; i < m_combinedLayers.size(); i++)
        {
            delete m_combinedLayers[i];
        }
        
    }

}
