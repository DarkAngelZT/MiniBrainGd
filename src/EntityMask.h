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

inline void BuildEntityMask(
    const float* players, int batch_size, int player_entity_num, int player_dim,
    const float* monsters, int monster_entity_num, int monster_dim,
    const float* bullets, int bullet_entity_num, int bullet_dim, float* mask) {
    const int total_entity_num =
        player_entity_num + monster_entity_num + bullet_entity_num;
    for (int batch = 0; batch < batch_size; ++batch) {
        float* batch_mask = mask + batch * total_entity_num;
        BuildEntityMask(players + batch * player_entity_num * player_dim,
                        1, player_entity_num, player_dim, batch_mask);
        BuildEntityMask(monsters + batch * monster_entity_num * monster_dim,
                        1, monster_entity_num, monster_dim,
                        batch_mask + player_entity_num);
        BuildEntityMask(bullets + batch * bullet_entity_num * bullet_dim,
                        1, bullet_entity_num, bullet_dim,
                        batch_mask + player_entity_num + monster_entity_num);
    }
}

} // namespace MiniMind
