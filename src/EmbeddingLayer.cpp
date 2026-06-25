// Implementation file for EmbeddingLayer template class.

#include "EmbeddingLayer.h"

namespace MiniBrain
{
    template<typename T>
    EmbeddingLayer<T>::EmbeddingLayer(int inSize, int feature_dim, int outfeature_dim)
        : FullyConnected<T>(inSize, (inSize / feature_dim) * outfeature_dim),
          m_in_feature_dim(feature_dim),
          m_out_feature_dim(outfeature_dim)
    {
        m_n_entities = inSize / feature_dim;
    }

    template<typename T>
    void EmbeddingLayer<T>::Init()
    {
        this->m_weight.resize(this->m_in_feature_dim,this->m_out_feature_dim);
        this->m_bias.resize(this->m_out_feature_dim);
        this->m_dw.resize(this->m_in_feature_dim,this->m_out_feature_dim);
        this->m_db.resize(this->m_out_feature_dim);
    }

    template<typename T>
    Matrix<T> EmbeddingLayer<T>::Forward(const Matrix<T>& InData)
    {
        Matrix<T> out(this->m_out_feature_dim*this->m_n_entities, InData.cols());
        const int nobs = InData.cols();

        if constexpr (std::is_same_v<T, AutoDiffVar>)
        {
            for (int i = 0; i < nobs; ++i)
            {
                Matrix<T> X_obs(m_in_feature_dim, m_n_entities);
                for (int s = 0; s < m_n_entities; ++s) {
                    X_obs.col(s) = InData.block(s * m_in_feature_dim, i, m_in_feature_dim, 1);
                }
                Matrix<T> embedded = this->m_weight.transpose() * X_obs;
                embedded.colwise() += this->m_bias;
                for (int s = 0; s < m_n_entities; ++s) 
                {
                    out.block(s * m_out_feature_dim, i, m_out_feature_dim, 1) = embedded.col(s);
                }
            }
        }
        else
        {
            // //扩展参数
            for (int i = 0; i < nobs; ++i)
            {
                Eigen::Map<const Matrix<Scalar>> X(InData.col(i).data(), m_in_feature_dim, m_n_entities);
                Matrix<T> embedded = this->m_weight.transpose() * X;
                embedded.colwise() += this->m_bias;
                Eigen::Map<Vector<T>>(out.col(i).data(), m_out_feature_dim * m_n_entities) = Eigen::Map<Vector<T>>(embedded.data(), m_out_feature_dim * m_n_entities);
            }
        }

        return out;
    }

    template<typename T>
    std::string EmbeddingLayer<T>::GetSubType() const
    {
        return "EmbeddingLayer";
    }
}
