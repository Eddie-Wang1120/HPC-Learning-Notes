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
    // Creating a Network Definition
    IBuilder* builder = createInferBuilder(logger);
    // The kEXPLICIT_BATCH flag is required in order to import models using the ONNX parser.
    uint32_t flag = 1U <<static_cast<uint32_t> (NetworkDefinitionCreationFlag::kEXPLICIT_BATCH);
    INetworkDefinition* network = builder->createNetworkV2(flag);

    // Importing a Model using the ONNX Parser
    IParser* parser = createParser(*network, logger);
    parser->parseFromFile("../chap3/resnet50-v2-7.onnx", static_cast<int>(ILogger::Severity::kWARNING));
    for (int32_t i = 0; i < parser->getNbErrors(); ++i)
    {
        std::cout << parser->getError(i)->desc() << std::endl;
    }

    // An important aspect of a TensorRT network definition is that it contains pointers to model
    // weights, which are copied into the optimized engine by the builder. Since the network was
    // created via the parser, the parser owns the memory occupied by the weights, and so the
    // parser object should not be deleted until after the builder has run.

    // Building an Engine
    IBuilderConfig* config = builder->createBuilderConfig();
    
    // Layer implementations often require a temporary workspace, and this parameter limits the
    // maximum size that any layer in the network can use.
    config->setMaxWorkspaceSize(1U << 20);

    IHostMemory* serialized_model = builder->buildSerializedNetwork(*network, *config);

    std::stringstream engine_file_stream;
    engine_file_stream.seekg(0, engine_file_stream.beg);
    engine_file_stream.write(static_cast<const char *>(serialized_model->data()),serialized_model->size());
    const std::string engine_file_path = "resnet50.engine";
    std::ofstream out_file(engine_file_path);
    assert(out_file.is_open());
    out_file << engine_file_stream.rdbuf();
    out_file.close();

    delete parser;
    delete network;
    delete config;
    delete builder;

    // Deserializing a Plan
    IRuntime* runtime = createInferRuntime(logger);
    ICudaEngine* engine = runtime->deserializeCudaEngine(serialized_model->data(), serialized_model->size());

    // The engine can then be saved to disk, and the buffer into which it was serialized can be deleted.
    delete serialized_model;

    // Performing Inference
    // to perform inference we will need to manage
    // additional state for intermediate activations
    IExecutionContext *context = engine->createExecutionContext();

    // An engine can have multiple execution contexts, allowing one set of weights to be used for
    // multiple overlapping inference tasks
    // except for dynamic shapes

    // must pass TensorRT buffers
    int32_t inputIndex = engine->getBindingIndex("data");
    int32_t outputIndex = engine->getBindingIndex("resnetv24_dense0_fwd");

    // set up a buffer array pointing to the input and output buffers on the GPU
    void* buffers[2];
}