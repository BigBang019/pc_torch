//
// Created by xingyu on 11/29/22.
//
#include "detach.h"
#include "sampling.h"
#include "ball_query.h"
#include "utils.h"

#include <set>


/**
 *
 * @param xyz   (B, N, 3)
 * @param ratio S = N * ratio
 * @return idx  (B, N-S)
 */
at::Tensor random_crop(at::Tensor xyz, const float ratio) {
    CHECK_CONTIGUOUS(xyz);
    CHECK_CUDA(xyz);
    CHECK_IS_FLOAT(xyz);

    const int B = xyz.size(0);
    const int S = xyz.size(1) * ratio;
    const int R = xyz.size(1) - S;

    at::Tensor out = torch::zeros({B, R},
                                  at::device(torch::kCPU).dtype(at::ScalarType::Int));

    at::Tensor center_idx = torch::randint(0, xyz.size(1), {B, 1},
                                           at::device(xyz.device()).dtype(at::ScalarType::Int)); // (B, 1)
    at::Tensor center = gather_points(
            xyz.permute({0, 2, 1}).contiguous(), center_idx
    ).permute({0, 2, 1}).contiguous(); // (B, 1, 3)

    at::Tensor cropped_idx = k_neighbor_query(center, xyz, S).squeeze(1).cpu(); // (B, S)

    int * cropped_idx_data = cropped_idx.data_ptr<int>();
    int * out_data = out.data_ptr<int>();
    for (int i = 0; i < B; ++i) { //B
        std::set<int> result;
        for (int j = 0; j < xyz.size(1); ++j) { //N
            result.emplace_hint(result.end(), j);
        }
        for (int j = 0; j < S; ++j) {
            result.erase(cropped_idx_data[i * S + j]);
        }
        auto it = result.begin();
        for (int j = 0; j < R; ++j) {
            auto tmp = *(it++);
            out_data[i * R + j] = tmp;
        }
    }
    return out.cuda();
}