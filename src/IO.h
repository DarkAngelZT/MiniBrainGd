#ifndef IO_H
#define IO_H

#include <fstream>
#include <godot_cpp/variant/string.hpp>
#include <optional>
#include <string>
#include "bitsery/bitsery.h"
#include "bitsery/ext/std_optional.h"

#include "../MiniBrain/Source/MiniBrain.h"

namespace godot {

    class AICheckpoint
    {
    public:
        MiniBrain::io::NetworkData preprocessNet;
        MiniBrain::io::NetworkData moveNet;
        MiniBrain::io::NetworkData shootNet;
        std::optional<MiniBrain::io::NetworkData> criticNet = std::nullopt;

        template<typename S>
        void serialize(S& s)
        {
            s.object(preprocessNet);
            s.object(moveNet);
            s.object(shootNet);
            s.ext(criticNet, bitsery::ext::StdOptional{});
        }
    };

    template<typename T>
    bool SaveCheckPoint(
        const std::string &path, 
        MiniBrain::Network<T>* preprocessNet,
        MiniBrain::Network<T>* moveNet,
        MiniBrain::Network<T>* shootNet,
        MiniBrain::Network<T>* criticNet
    ){
        std::ofstream stream(path, std::ios::binary);
        if (!stream.is_open()) {
            return false;
        }
        bitsery::Serializer<bitsery::OutputBufferedStreamAdapter> ser(stream);
        auto pack = [&](const std::vector<std::vector<MiniBrain::Scalar>>& data) -> MiniBrain::io::NetworkData {
            MiniBrain::io::NetworkData nd;
            for(auto& d:data)
            {
                MiniBrain::io::SingleParamData spd;
                spd.data = d;
                nd.data.push_back(spd);
            }
            return nd;
        };

        AICheckpoint ckpt;
        ckpt.preprocessNet = pack(preprocessNet->GetParameters());
        ckpt.moveNet = pack(moveNet->GetParameters());
        ckpt.shootNet = pack(shootNet->GetParameters());

        if constexpr (std::is_same_v<T, MiniBrain::AutoDiffVar>)
        {
            //训练模式才会有critic
            ckpt.criticNet = pack(criticNet->GetParameters());
        }

        ser.object(ckpt);
        ser.adapter().flush();
        stream.close();
        return true;
    }

    template<typename T>
    bool LoadCheckPoint(
        const std::string &path, 
        MiniBrain::Network<T>* preprocessNet,
        MiniBrain::Network<T>* moveNet,
        MiniBrain::Network<T>* shootNet,
        MiniBrain::Network<T>* criticNet
    ){
        std::ifstream stream(path, std::ios::binary);
        if (!stream.is_open()) {
            return false;
        }
        bitsery::Deserializer<bitsery::InputStreamAdapter> des(stream);
        auto unpack = [&](const MiniBrain::io::NetworkData& nd , std::vector<std::vector<MiniBrain::Scalar>>& data){
            for(auto& d:nd.data)
            {
                data.push_back(d.data);
            }
        };
        AICheckpoint ckpt;
        des.object(ckpt);        
        stream.close();

        std::vector<std::vector<MiniBrain::Scalar>> preproceParam;
        std::vector<std::vector<MiniBrain::Scalar>> moveParam;
        std::vector<std::vector<MiniBrain::Scalar>> shootParam;

        unpack(ckpt.preprocessNet, preproceParam);
        unpack(ckpt.moveNet, moveParam);
        unpack(ckpt.shootNet, shootParam);
        preprocessNet->SetParameters(preproceParam);
        moveNet->SetParameters(moveParam);
        shootNet->SetParameters(shootParam);

        if constexpr (std::is_same_v<T, MiniBrain::AutoDiffVar>)
        {
            //训练模式才会有critic
            if(ckpt.criticNet.has_value())
            {
                std::vector<std::vector<MiniBrain::Scalar>> criticParam;
                unpack(ckpt.criticNet.value(), criticParam);
                criticNet->SetParameters(criticParam);
            }
        }
        return true;
    }

}

#endif // IO_H
