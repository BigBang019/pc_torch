cmake_minimum_required (VERSION 3.23.0)

######### # essential to make it find nvcc@11.4 #########
set(CMAKE_CUDA_COMPILER /usr/local/cuda/bin/nvcc)
set(CUDA_TOOLKIT_ROOT_DIR /usr/local/cuda/)
#set(CMAKE_CXX_FLAGS "-D_GLIBCXX_USE_CXX11_ABI=0")

project (PointConv LANGUAGES CUDA CXX)

######### CUDA LIB #########
find_package(CUDA 11.4 REQUIRED)

######### CUDNN PATH FOR TORCH #########
set(CUDNN_INCLUDE_DIR /data_HDD/zhuxingyu/anaconda3/envs/p11/include)
set(CUDNN_INCLUDE_PATH /data_HDD/zhuxingyu/anaconda3/envs/p11/include)
set(CUDNN_LIB_DIR /data_HDD/zhuxingyu/anaconda3/envs/p11/lib)
set(CUDNN_PATH /data_HDD/zhuxingyu/anaconda3/envs/p11/liblibcudnn.so)
set(CUDNN_LIBRARY_PATH /data_HDD/zhuxingyu/anaconda3/envs/p11/lib)

######### TORCH #########
set(CMAKE_PREFIX_PATH ${CMAKE_PREFIX_PATH} "libtorch")
find_package(Torch REQUIRED)

######### SET TARGET #########
add_executable(PointConv main.cu)
set_target_properties(PointConv PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
target_compile_features(PointConv PUBLIC cxx_std_11)

######### INCLUDE #########
include_directories("libtorch/include/")
include_directories("libtorch/include/torch/csrc/api/include/")
include_directories("/data_HDD/zhuxingyu/anaconda3/envs/p11/include/python3.8/")
include_directories("${PROJECT_SOURCE_DIR}/_ext/include")

######### SUBDIR #########
add_subdirectory(_ext)
target_link_libraries(PointConv _ext)

######### ADD LIBRARY #########
file(GLOB LIBTORCH_LIBS "libtorch/lib/*.a" "libtorch/lib/*.so")
target_link_libraries(PointConv ${LIBTORCH_LIBS} ${CUDA_LIBRARIES} ${TORCH_LIBRARIES})