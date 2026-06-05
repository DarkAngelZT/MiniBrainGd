#include "StatePooling.h"

namespace MiniBrain {
    template<typename T>
    StatePooling<T>::StatePooling(int inSize, int outSize)
        : Layer<T>(inSize, outSize), m_featureDim(outSize)
    {
        Init();
    }

    template<typename T>
    void StatePooling<T>::Init()
    {
    }

    template<typename T>
    void StatePooling<T>::Init(const Scalar& mu, const Scalar& sigma, Random& RNG)
    {
    }

    template<typename T>
    Matrix<T> StatePooling<T>::Forward(const Matrix<T>& InData)
    {
        int B = InData.cols();// 计算当前时刻有几个环境
        int n_entities = InData.rows() / m_featureDim; // 每个环境有多少个实体
        
        // 输出矩阵：每个环境输出一个 2*D 维度的向量（Mean和Max拼接）
        Matrix<T> outData(2 * m_featureDim, B); 

        if constexpr (std::is_same_v<T, AutoDiffVar>)
        {
            for (int b = 0; b < B; ++b) 
            {
                Matrix<T> env_col = InData.col(b);
                for (int j = 0; j < m_featureDim; ++j) 
                {
                    // 必须显式使用 Scalar(0.0) 初始化以确保在全局 Tape 上注册累加器
                    AutoDiffVar sum_val = AutoDiffVar(0.0);
                    
                    // 显式以第一个实体对应的特征值作为最大值的起点
                    // 对应一维列向量的索引为：实体 id * 特征维度 + 特征 id
                    AutoDiffVar max_val = env_col(j); // 第一个实体的第 j 个特征值
                    sum_val += max_val;

                    for (int e = 1; e < n_entities; ++e) 
                    {
                        AutoDiffVar current_val = env_col(e * m_featureDim + j);
                        sum_val += current_val;

                        if (current_val.expr->val > max_val.expr->val) { 
                            max_val = current_val; 
                        }
                    }

                    // 计算均值（由于 sum_val 是 var，除以常量会自动生成一个除法/乘法表达式节点流向均值）
                    AutoDiffVar mean_val = sum_val / n_entities;

                    // 写入输出列的对应特征行位置
                    outData(j, b)                = mean_val; // 均值部分放上半区
                    outData(m_featureDim + j, b) = max_val;  // 最大值部分放下半区
                }
            }
        }
        else
        {
            for (int b = 0; b < B; ++b) {
                // 切出第 b 个环境的 [N x D] 矩阵
                Eigen::Map<const Matrix<Scalar>> X(InData.col(b).data(), m_featureDim, n_entities);

                // 1. 计算平均池化 (D x 1)
                Vector<Scalar> mean_val = X.rowwise().mean();

                // 2. 计算最大池化 (D x 1) 并记录索引
                Vector<Scalar> max_val(m_featureDim);
                for (int j = 0; j < m_featureDim; ++j) {
                    max_val(j) = X.row(j).maxCoeff();// 记录第 b 个环境第 j 个特征的最大值在第几行
                }

                // 3. 拼接 Mean 和 Max 写入输出的一行 (1 x 2D)
                outData.block(0, b, m_featureDim, 1) = mean_val;
                outData.block(m_featureDim, b, m_featureDim, 1) = max_val;
            }
            return outData;
        }       
    }

    template<typename T>
    void StatePooling<T>::Backward(T& Loss)
    {
    }

    template<typename T>
    void StatePooling<T>::Update(Optimizer<Scalar>& opt)
    {
    }

    template<typename T>
    std::vector<Scalar> StatePooling<T>::GetParameters() const
    {
        return {};
    }

    template<typename T>
    std::string StatePooling<T>::GetSubType() const
    {
        return "StatePooling";
    }

    template class StatePooling<Scalar>;
    template class StatePooling<AutoDiffVar>;
} // namespace MiniBrain
