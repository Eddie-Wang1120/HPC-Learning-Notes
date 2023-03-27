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

    // the option to do so must be specified when building
    config->setFlag(BuilderFlag::kREFIT);

    IHostMemory* serialized_model = builder->buildSerializedNetwork(*network, *config);

    delete parser;
    delete network;
    delete config;
    delete builder;

    // Deserializing a Plan
    IRuntime* runtime = createInferRuntime(logger);
    ICudaEngine* engine = runtime->deserializeCudaEngine(serialized_model->data(), serialized_model->size());

    IRefitter* refitter = createInferRefitter(*engine, logger);

    // to update the kernel weights for a convolution layer
    // named “MyLayer”
    Weights newWeights = ...;
    refitter->setWeights("MyLayer", WeightsRole::kKERNEL, newWeights);
    refitter->setNamedWeights("MyWeights", newWeights);

    // This typically requires two calls to IRefitter::getMissing , first to get the number of
    // weights objects that must be supplied, and second to get their layers and roles
    const int32_t n = refitter->getMissing(0, nullptr, nullptr);
    std::vector<const char*> layerNames(n);
    std::vector<WeightsRole> weightsRoles(n);
    refitter->getMissing(n, layerNames.data(), weightsRoles.data());

    bool success = refitter->refitCudaEngine();
    assert(succes);

    delete refitter;
}   