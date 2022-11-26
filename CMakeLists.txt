cmake_minimum_required (VERSION 3.23.0)
project (PointConv LANGUAGES CUDA)

add_executable(PointConv main.cu)

set_target_properties(PointConv PROPERTIES CUDA_SEPARABLE_COMPILATION ON)

# include
include_directories("libtorch/include/")
include_directories("libtorch/include/torch/csrc/api/include/")
include_directories("/data_HDD/zhuxingyu/anaconda3/envs/p11/include/python3.8/")
include_directories("${PROJECT_SOURCE_DIR}/_ext/include")
# subdir
add_subdirectory(_ext)
target_link_libraries(PointConv _ext)

file(GLOB LIBTORCH_LIBS "libtorch/lib/*.a" "libtorch/lib/*.so")
target_link_libraries(PointConv ${LIBTORCH_LIBS})