#pragma once

#include "../MiniBrain/Source/Layer.h"
#include "../MiniBrain/Source/Optimizer.h"
#include "../MiniBrain/Source/Random.h"
#include <vector>

namespace MiniBrain {
    template<typename T>
    class StatePooling: public Layer<T>
    {
    private:
        Eigen::MatrixXi max_indices;
        int m_featureDim;
    public:
        StatePooling(int inSize, int outSize);

        virtual void Init() override;

        virtual void Init(const Scalar& mu, const Scalar& sigma, Random& RNG) override;

        virtual Matrix<T> Forward(const Matrix<T>& InData) override;

        virtual void Backward(T& Loss) override;

        virtual void Update(Optimizer<Scalar>& opt) override;

        virtual bool HasParameters() const override { return false; }

        virtual std::vector<Scalar> GetParameters() const override;

        virtual std::string GetSubType() const override;
    };
} // namespace MiniBrain
