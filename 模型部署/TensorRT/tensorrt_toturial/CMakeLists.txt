project(tensorrt)
include_directories(${CMAKE_CURRENT_SOURCE_DIR})
include_directories(/usr/local/cuda/include)

find_package(CUDA)
if( NOT CUDA_FOUND )
        message(STATUS "CUDA Not Found. Project will not be build.")
endif( CUDA_FOUND )

add_executable(test chap3/build_phase.cc)
add_executable(test_cache chap6/The_Timing_Cache.cc)
cuda_add_executable(test_cuda chap3/deserialized.cu)

target_link_libraries(
    test
    PUBLIC nvinfer
    PUBLIC nvonnxparser
    PUBLIC cudart
)

target_link_libraries(
    test_cuda
    PUBLIC nvinfer
    PUBLIC nvonnxparser
    PUBLIC cudart
)

target_link_libraries(
    test_cache
    PUBLIC nvinfer
    PUBLIC nvonnxparser
    PUBLIC cudart
)