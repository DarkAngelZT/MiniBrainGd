#pragma once

#include <godot_cpp/classes/object.hpp>
#include "ShooterNetwork.h"

namespace godot {
    class AIAgent: public Object
    {
    protected:
        ShooterNetwork* network;
    public:
        AIAgent(/* args */);
        ~AIAgent();

        void Init();

        PackedFloat32Array ProcessSensorData(const PackedFloat32Array& data);

        void PushTrainingData();

        void Train();
    };    
}