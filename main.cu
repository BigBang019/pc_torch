//
// Created by xingyu on 11/26/22.
//
#include "detach.h"

int main(){
    const int B = 4;
    const int N = 10;
    at::Tensor pc = torch::randn({B, N, 3},
                                 at::dtype(at::ScalarType::Float)).cuda();
    at::Tensor idx = random_crop(pc, 0.3);
    std::cout << idx.size(0) << " " << idx.size(1) << std::endl;
    return 0;
}