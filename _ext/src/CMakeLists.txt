project(_ext LANGUAGES CUDA)
set(CUDA_TOOLKIT_ROOT_DIR /usr/local/cuda)

file(GLOB CUDA_FILES "*.cu")
#file(GLOB CXX_FILES "*.cpp")
# cmake will not comiple cu files without this statement
add_library(_ext ${CUDA_FILES} ${CXX_FILES})

#target_compile_features(_ext PUBLIC cxx_std_11)
set_target_properties(_ext PROPERTIES CUDA_SEPARABLE_COMPILATION ON)