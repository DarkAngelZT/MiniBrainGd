#pragma once

#include <godot_cpp/classes/object.hpp>

namespace godot {
    class AIAgent: public Object
    {
    protected:
        /* data */
    public:
        AIAgent(/* args */);
        ~AIAgent();

        void Init();

        void ProcessSensorData();

        void PushTrainingData();

        void Train();
    };    
}