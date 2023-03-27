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
    const std::string engine_file_path = "resnet50.engine";
    std::stringstream engine_file_stream;
    engine_file_stream.seekg(0, engine_file_stream.beg);
    std::ifstream ifs(engine_file_path);
    engine_file_stream << ifs.rdbuf();
    ifs.close();

    engine_file_stream.seekg(0, std::ios::end);
    const int model_size = engine_file_stream.tellg();
    engine_file_stream.seekg(0, std::ios::beg);
    void *model_mem = malloc(model_size);
    engine_file_stream.read(static_cast<char *>(model_mem), model_size);

    IRuntime *runtime = createInferRuntime(logger);
    ICudaEngine *engine = runtime->deserializeCudaEngine(model_mem, model_size);    

    nvinfer1::IExecutionContext *context = engine->createExecutionContext();

    void *buffers[2];
    // 获取模型输入尺寸并分配GPU内存
    Dims input_dim = engine->getBindingDimensions(0);
    int input_size = 1;
    for (int j = 0; j < input_dim.nbDims; ++j) {
        input_size *= input_dim.d[j];
    }
    cudaMalloc(&buffers[0], input_size * sizeof(float));

    // 获取模型输出尺寸并分配GPU内存
    Dims output_dim = engine->getBindingDimensions(1);
    int output_size = 1;
    for (int j = 0; j < output_dim.nbDims; ++j) {
        output_size *= output_dim.d[j];
    }
    cudaMalloc(&buffers[1], output_size * sizeof(float));

    // 给模型输出数据分配相应的CPU内存
    float *output_buffer = new float[output_size]();

    // cudaStream_t stream;
    // cudaStreamCreate(&stream);
    // // 拷贝输入数据
    // cudaMemcpyAsync(buffers[0], input_blob,input_size * sizeof(float),
    //               cudaMemcpyHostToDevice, stream);
    // // 执行推理
    // context->enqueueV2(buffers, stream, nullptr);
    // // 拷贝输出数据
    // cudaMemcpyAsync(output_buffer, buffers[1],output_size * sizeof(float),
    //               cudaMemcpyDeviceToHost, stream);

    // cudaStreamSynchronize(stream);
}