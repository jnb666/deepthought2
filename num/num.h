#ifndef NUM_H
#define NUM_H

#include <stdint.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/time.h>
#include <mkl.h>
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <cudnn.h>

#define QUEUE_SIZE 128
#define EPS 1e-7f

#define I32 0
#define F32 1

#define FP(a) (float*)(a)
#define IP(a) (int*)(a)

#define clamp(x, min, max)  ((x) < (min) ? (min) : ((x) > (max) ? (max) : (x)))

enum {COPY, COPY_TO_DEVICE, COPY_TO_HOST, COPY_COL, TILE0, TILE1, 
	FILL, NEQ, ONEHOT, UNHOT, AXPY, TRANS, SUM, GEMV, GEMM, MUL_ELEM,
	SIGMOID, SIGMOID_D, TANH, TANH_D, RELU, RELU_D, 
	QUAD_LOSS, SOFTMAX, SOFTMAX_LOSS, MKL_DNN_EXECUTE, CUDNN_EXECUTE};

enum{CUDNN_ACTIV_FPROP=CUDNN_EXECUTE, CUDNN_ACTIV_BPROP, 
	CUDNN_DROPOUT_FPROP, CUDNN_DROPOUT_BPROP,
	CUDNN_BNORM_FPROP_INFER, CUDNN_BNORM_FPROP_TRAIN, CUDNN_BNORM_BPROP,
	CUDNN_CONV_FPROP, CUDNN_CONV_FPROP_BIAS, 
	CUDNN_CONV_BPROP_DATA, CUDNN_CONV_BPROP_FILTER, CUDNN_CONV_BPROP_BIAS,
	CUDNN_POOL_FPROP, CUDNN_POOL_BPROP};

typedef struct args {
	int     op;
	float   msec;
	int     i[8];
	float   f[4];
	void*   p[10];
	void*   desc;
} Args;

typedef struct stream {
	cudaStream_t	stream;
	cublasHandle_t	blas;
	cudnnHandle_t	cudnn;
} Stream;

typedef struct events {
	cudaEvent_t start[QUEUE_SIZE];
	cudaEvent_t end[QUEUE_SIZE];
} Events;

void initEvents(Events* e);

int execCPU(int nargs, Args** buffer, dnnError_t* error);

int execCPUProfile(int nargs, Args** buffer, dnnError_t* error);

int execGPU(int nargs, Args** buffer, Stream* stream, cudaError_t* error, cublasStatus_t* blasError, cudnnStatus_t* dnnError);

int execGPUProfile(int nargs, Args** buffer, Stream* stream, cudaError_t* error, cublasStatus_t* blasError, cudnnStatus_t* dnnError, Events* e);

#endif // NUM_H