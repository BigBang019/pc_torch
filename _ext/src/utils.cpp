//
// Created by xingyu on 12/14/22.
//

#include "utils.h"

void build_graph_from_triangle_kernel_wrapper(int B, int N, int M, const int *__restrict__ polygons,
                                              int *__restrict__ header,
                                              int *__restrict__ cnt,
                                              int *__restrict__ v,
                                              int *__restrict__ next);

void merge_idx_kernel_wrapper(int B, int N, int S, const int *__restrict to, const int *__restrict__ from, int *__restrict__ output);

/**
 * Some polygons are padded with -1 in the end
 * Will Skip all -1
 * @param xyz
 * @param polygons
 * @return
 */
std::vector<at::Tensor> build_graph_from_triangle(at::Tensor xyz, at::Tensor polygons) {
    CHECK_CONTIGUOUS(polygons);
    CHECK_CUDA(polygons);
    CHECK_IS_INT(polygons);

    const int B = xyz.size(0);
    const int N = xyz.size(1);
    const int M = polygons.size(1);

    at::Tensor header = torch::full({B, N}, -1, at::device(xyz.device()).dtype(at::ScalarType::Int));
    at::Tensor cnt = torch::zeros({B}, at::device(xyz.device()).dtype(at::ScalarType::Int));
    at::Tensor v = torch::zeros({B, 6*M}, at::device(xyz.device()).dtype(at::ScalarType::Int));
    at::Tensor next = torch::full({B, 6*M}, -1, at::device(xyz.device()).dtype(at::ScalarType::Int));

    build_graph_from_triangle_kernel_wrapper(B, N, M, polygons.data_ptr<int>(), header.data_ptr<int>(),
                                                  cnt.data_ptr<int>(), v.data_ptr<int>(), next.data_ptr<int>());

    return std::vector<at::Tensor>{header, cnt, v, next};
}

at::Tensor merge_idx(at::Tensor to, at::Tensor from) {
    CHECK_CONTIGUOUS(to);
    CHECK_CUDA(to);
    CHECK_IS_INT(to);
    CHECK_CONTIGUOUS(from);
    CHECK_CUDA(from);
    CHECK_IS_INT(from);

    const int B = to.size(0);
    const int N = to.size(1);
    const int S = to.size(2);

    at::Tensor output = torch::zeros({B, N, S}, at::device(from.device()).dtype(at::ScalarType::Int));

    merge_idx_kernel_wrapper(B, N, S, to.data_ptr<int>(), from.data_ptr<int>(), output.data_ptr<int>());

    return output;
}