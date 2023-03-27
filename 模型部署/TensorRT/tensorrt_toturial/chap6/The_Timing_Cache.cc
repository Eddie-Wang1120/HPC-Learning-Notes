#include "NvInfer.h"
#include "NvOnnxParser.h"
#include <iostream>
#include <stdint.h>
#include <assert.h>
#include <fstream>
#include <sstream>

using namespace nvinfer1;
using namespace nvonnxparser;

class Logger : public ILogger
{
    void log(Severity severity, const char* msg) noexcept override
    {
        if (severity != Severity::kINFO)
            std::cout << msg << std::endl;
    }
}logger;

int main(){
    IBuilder *builder = createInferBuilder(logger);
    uint32_t flag = 1U <<static_cast<uint32_t> (NetworkDefinitionCreationFlag::kEXPLICIT_BATCH);
    INetworkDefinition* network = builder->createNetworkV2(flag);

    IParser* parser = createParser(*network, logger);
    parser->parseFromFile("../chap3/resnet50-v2-7.onnx", static_cast<int>(ILogger::Severity::kWARNING));
    for (int32_t i = 0; i < parser->getNbErrors(); ++i){
        std::cout << parser->getError(i)->desc() << std::endl;
    }

    IBuilderConfig* config = builder->createBuilderConfig();
    config->setMaxWorkspaceSize(1U << 20);

    IHostMemory* serialized_model = builder->buildSerializedNetwork(*network, *config);

    // load a serialized cache from a buffer via IBuilderConfig::createTimingCache
    ITimingCache* cache = config->createTimingCache(serialized_model->data(), serialized_model->size());
     
    // then attach the cache to a builder configuration before building.
    config->setTimingCache(*cache, false);

    // can be serialized for use with another builder
    IHostMemory* serializedCache = cache->serialize();

}