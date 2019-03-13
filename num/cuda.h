#ifndef CUDA_H
#define CUDA_H

#include <cuda_runtime.h>

void cuda_fill_f(cudaStream_t stream, float* a, float val, int n);

void cuda_fill_i(cudaStream_t stream, int* a, float val, int n);

void cuda_tile0(cudaStream_t stream, float* dst, float* src, int rows, int cols);

void cuda_tile1(cudaStream_t stream, float* dst, float* src, int rows, int cols);

void cuda_sum_f(cudaStream_t stream, float* in, float* total, int n);

void cuda_sum_i(cudaStream_t stream, int* in, float* total, int n);

void cuda_neq(cudaStream_t stream, int* a, int* b, int* res, int n);

void cuda_mul_elem(cudaStream_t stream, float* a, float* b, float* res, int n);

void cuda_div_elem(cudaStream_t stream, float* a, float* b, float* res, float eps, int n);

void cuda_square(cudaStream_t stream, float* x, float* y, int n);

void cuda_sqrt(cudaStream_t stream, float* x, float* y, int n);

void cuda_onehot(cudaStream_t stream, int* y, float* y_one_hot, int n, int classes);

void cuda_unhot(cudaStream_t stream, float* y_one_hot, int* y, int n, int classes);

void cuda_quad_loss(cudaStream_t stream, float* y, float* y_pred, float* res, int n);

void cuda_softmax(cudaStream_t stream, float* x, float* res, int classes, int n);

void cuda_softmax_loss(cudaStream_t stream, float* y, float* y_pred, float* res, int classes, int n);

#endif // CUDA_H