project(tensorrt_samples)
set(CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/cmake)
set(CMAKE_CXX_STANDARD 11)

include_directories(${CMAKE_CURRENT_SOURCE_DIR})
include_directories(/usr/local/cuda/include)

find_package(CUDA)
if( NOT CUDA_FOUND )
        message(STATUS "CUDA Not Found. Project will not be build.")
endif( CUDA_FOUND )

