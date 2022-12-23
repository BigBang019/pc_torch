//
// Created by xingyu on 12/14/22.
//

#include "cuda_utils.h"


__device__ bool check_exist(int from, int to,
                            int *__restrict__ header,
                            int *__restrict__ cnt,
                            int *__restrict__ v,
                            int *__restrict__ next) {
    for (int i = header[from]; i != -1; i = next[i]) {
        if (v[i] == to) {
            return true;
        }
    }
    return false;
}

__device__ void add_edge(int from, int to,
                         int *__restrict__ header,
                         int *__restrict__ cnt,
                         int *__restrict__ v,
                         int *__restrict__ next) {
    if (check_exist(from, to, header, cnt, v, next)) {
        return;
    }
    int c = cnt[0]++;
    v[c] = to;
    next[c] = header[from];
    header[from] = c;
}
/**
 *
 * @param B batch size
 * @param N number of vertices
 * @param M number of polygons
 * @param polygons (B, M, 3)
 * @param header (B, N)
 * @param cnt (B)
 * @param v (B, 6M)
 * @param next (B, 6M)
 */
__global__ void build_graph_from_triangle_kernel(int B, int N, int M, const int *__restrict__ polygons,
                                   int *__restrict__ header,
                                   int *__restrict__ cnt,
                                   int *__restrict__ v,
                                   int *__restrict__ next) {
    for (int b = blockIdx.x; b < B; b += gridDim.x) {
        // has to be sequential execution
        for (int m = 0; m < M; ++m) {
            int x = polygons[(b * M + m) * 3], y = polygons[(b * M + m) * 3 + 1], z = polygons[(b * M + m) * 3 + 2];
            if (x == -1) {
                break;
            }
            add_edge(x, y, header+b*N, cnt+b, v+b*6*M , next+b*6*M);
            add_edge(y, x, header+b*N, cnt+b, v+b*6*M , next+b*6*M);
            add_edge(x, z, header+b*N, cnt+b, v+b*6*M , next+b*6*M);
            add_edge(z, x, header+b*N, cnt+b, v+b*6*M , next+b*6*M);
            add_edge(y, z, header+b*N, cnt+b, v+b*6*M , next+b*6*M);
            add_edge(z, y, header+b*N, cnt+b, v+b*6*M , next+b*6*M);
        }
    }
}

/**
 *
 * @param B batch size
 * @param N number of vertices
 * @param M number of polygons
 * @param polygons (B, M, 3)
 * @param header (B, N)
 * @param cnt (B)
 * @param v (B, 6M)
 * @param next (B, 6M)
 */
void build_graph_from_triangle_kernel_wrapper(int B, int N, int M, const int *__restrict__ polygons,
                                int *__restrict__ header,
                                int *__restrict__ cnt,
                                int *__restrict__ v,
                                int *__restrict__ next) {
    cudaStream_t stream = at::cuda::getCurrentCUDAStream();

    build_graph_from_triangle_kernel<<<B, 1, 0, stream>>>(B, N, M, polygons, header, cnt, v, next);

    CUDA_CHECK_ERRORS();
}

/**
 *
 * @param B
 * @param N
 * @param S
 * @param to (B, N, S)
 * @param from
 * @param output
 */
__global__ void merge_idx_kernel(int B, int N, int S, const int *__restrict to, const int *__restrict__ from, int *__restrict__ output) {
    for (int b = blockIdx.x; b < B; b+=gridDim.x) {
        for (int n = blockIdx.y; n < N; n+=gridDim.y) {
            for (int s = threadIdx.x; s < S; s+=blockDim.x) {
                output[(b*N+n)*S+s] = to[(b*N+n)*S+s]==-1?from[(b*N+n)*S+s]:to[(b*N+n)*S+s];
            }
        }
    }
}

void merge_idx_kernel_wrapper(int B, int N, int S, const int *__restrict to, const int *__restrict__ from, int *__restrict__ output) {
    cudaStream_t stream = at::cuda::getCurrentCUDAStream();

    merge_idx_kernel<<<dim3(B, N, 1), opt_n_threads(S), 0, stream>>>(B, N, S, to, from, output);

    CUDA_CHECK_ERRORS();
}