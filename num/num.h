#ifndef NUM_H
#define NUM_H

#include <stdint.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/time.h>
#ifdef USE_MKL
#include <mkl.h>
#else
#include <cblas.h>
#endif

#define EPS 1e-7f

#define I32 0
#define F32 1

#define FP(a) (float*)(a)
#define IP(a) (int*)(a)

#define clamp(x, min, max)  ((x) < (min) ? (min) : ((x) > (max) ? (max) : (x)))

enum {COPY, COPY_ROW, COPY_COL, TILE0, TILE1, FILL, NEQ, ONEHOT, UNHOT, SCALE, AXPY, TRANS, SUM, 
	GEMV, GEMM, SIGMOID, SIGMOID_D, TANH, TANH_D, RELU, RELU_D, QUAD_LOSS, SOFTMAX, SOFTMAX_LOSS, 
	DNN_EXECUTE};

typedef struct args {
	int     op;
	int     i[8];
	float   f[4];
	void*   p[4];
	int64_t usec;
	void*   desc;
} Args;

void set_num_threads(int n);

dnnError_t execCPU(int nargs, Args** buffer, int* ix);

dnnError_t execCPUProfile(int nargs, Args** buffer, int* ix);

#endif // NUM_H