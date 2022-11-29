//
// Created by xingyu on 11/29/22.
//

#ifndef POINTCONV_DETACH_H
#define POINTCONV_DETACH_H

#pragma once
#include <torch/extension.h>

at::Tensor random_crop(at::Tensor xyz, const float ratio);

#endif //POINTCONV_DETACH_H
