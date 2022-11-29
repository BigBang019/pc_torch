// Copyright (c) Facebook, Inc. and its affiliates.
// 
// This source code is licensed under the MIT license found in the
// LICENSE file in the root directory of this source tree.

#ifndef POINTCONV_UTIL_H
#define POINTCONV_UTIL_H

#pragma once
#include <ATen/cuda/CUDAContext.h>
#include <torch/extension.h>
#include <iostream>

#define CHECK_CUDA(x)                                          \
  do {                                                         \
    TORCH_CHECK(x.type().is_cuda(), #x " must be a CUDA tensor"); \
  } while (0)

#define CHECK_CONTIGUOUS(x)                                         \
  do {                                                              \
    TORCH_CHECK(x.is_contiguous(), #x " must be a contiguous tensor"); \
  } while (0)

#define CHECK_IS_INT(x)                              \
  do {                                               \
    TORCH_CHECK(x.scalar_type() == at::ScalarType::Int, \
             #x " must be an int tensor");           \
  } while (0)

#define CHECK_IS_FLOAT(x)                              \
  do {                                                 \
    TORCH_CHECK(x.scalar_type() == at::ScalarType::Float, \
             #x " must be a float tensor");            \
  } while (0)


#define DEBUG(...) logger(#__VA_ARGS__, __VA_ARGS__)
template<typename ...Args>
void logger(std::string vars, Args&&... values) {
    std::cout << vars << " = ";
    std::string delim = "";
    (..., (std::cout << delim << values, delim = ", "));
    std::cout << std::endl;
}
//#define DEBUG(x) do { std::cout << #x << "=" << x; } while (0)

#endif