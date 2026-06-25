#pragma once

#include "../MiniBrain/Source/Layers/FullyConnected.h"
#include "../MiniBrain/Source/Optimizer.h"
#include "../MiniBrain/Source/Random.h"
#include <vector>
#include <string>

namespace MiniBrain
{
    template<typename T>
    class EmbeddingLayer : public FullyConnected<T>
    {
    protected:
        int m_in_feature_dim;
        int m_out_feature_dim;
        int m_n_entities;
    public:
        EmbeddingLayer(int inSize, int feature_dim, int outfeature_dim);

        virtual ~EmbeddingLayer() override = default;

        virtual void Init() override;

        virtual Matrix<T> Forward(const Matrix<T>& InData) override;

        virtual std::string GetSubType() const override;
    };
}
