#pragma once

namespace MiniMind {

inline void BuildEntityMask(const float* states, int batch_size,
                            int entity_num, int feature_dim, float* mask) {
    for (int batch = 0; batch < batch_size; ++batch) {
        for (int entity = 0; entity < entity_num; ++entity) {
            const int entity_offset =
                (batch * entity_num + entity) * feature_dim;
            bool has_nonzero_feature = false;
            for (int feature = 0; feature < feature_dim; ++feature) {
                if (states[entity_offset + feature] != 0.0f) {
                    has_nonzero_feature = true;
                    break;
                }
            }
            mask[batch * entity_num + entity] =
                has_nonzero_feature ? 1.0f : 0.0f;
        }
    }
}

} // namespace MiniMind
