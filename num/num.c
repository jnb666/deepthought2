// Cgo interface routines
#include "num.h"

void array_copy_row(int* dst, int* src, int row, int rows, int cols) {
	for (int i = 0; i < cols; ++i) dst[row+i*rows] = src[i];
}

void array_fill_f(float* a, float val, int n) {
	for (int i = 0; i < n; ++i) a[i] = val;
}

void array_fill_i(int* a, float val, int n) {
	int ival = val;
	for (int i = 0; i < n; ++i) a[i] = ival;
}			

void array_neq(int* a, int* b, int* res, int n) {
	for (int i = 0; i < n; ++i) res[i] = (a[i] != b[i]);
}

void array_tile0(void* dst, void* src, int vlen, int cols) {
	for (int i = 0; i < cols; ++i) {
		memcpy(dst+i*vlen, src, vlen);				
	}
}

void array_tile1(float* dst, float* src, int vlen, int rows) {
	for (int j = 0; j < vlen; ++j) {
		array_fill_f(dst+j*rows, src[j], rows);
	}
}

void array_sum_f(float* a, int n, float scale, float* res) {
	float sum = 0.f;
	for (int i = 0; i < n; ++i) sum += a[i];
	*res = scale * sum;
}

void array_sum_i(int* a, int n, float scale, float* res) {
	int sum = 0;
	for (int i = 0; i < n; ++i) sum += a[i];
	*res = scale * (float)sum;
}

void onehot(int* y, float* y_one_hot, int n, int classes) {
	if (classes == 1) {
		for (int i = 0; i < n; ++i) {
			y_one_hot[i] = (float) y[i];
		}
	} else {
		memset(y_one_hot, 0, 4*n*classes);
		for (int i = 0; i < n; ++i) {
			y_one_hot[y[i] + i*classes] = 1.f;
		}
	}
}

void unhot(float* y_one_hot, int* y, int n, int classes) {
	if (classes == 1) {
		for (int i = 0; i < n; ++i) {
			y[i] = y_one_hot[i] > 0.5;
		}
	} else {
		for (int col = 0; col < n; ++col) {
			int ix = 0;
			int base = col*classes;
			float max = y_one_hot[base];
			for (int i = 1; i < classes; ++i) {
				if (y_one_hot[base+i] > max) {
					max = y_one_hot[base+i];
					ix = i;
				}
			}
			y[col] = ix;
		}
	}
}

void sigmoid_a(float* x, float* y, int n) {
	for (int i = 0; i < n; ++i) {
		y[i] = 1.f / (1.f + expf(-x[i]));
	}
}

void sigmoid_d(float* x, float* grad, float* y, int n) {
	for (int i = 0; i < n; ++i) {
		float s = 1.f / (1.f + expf(-x[i]));
		y[i] = grad[i] * s * (1.f-s);
	}
}

void tanh_a(float* x, float* y, int n) {
	for (int i = 0; i < n; ++i) {
		y[i] = tanhf(x[i]);
	}
}

void tanh_d(float* x, float* grad, float* y, int n) {
	for (int i = 0; i < n; ++i) {
		float e = expf(2.f*x[i]);
		y[i] = grad[i] * (e-1) / (e+1);
	}
}

void relu_a(float* x, float* y, int n) {
	for (int i = 0; i < n; ++i) {
		y[i] = (x[i] > 0.f) ? x[i] : 0.f;
	}
}

void relu_d(float* x, float* grad, float* y, int n) {
	for (int i = 0; i < n; ++i) {
		y[i] = grad[i] * (float)(x[i] > 0.f);
	}
}

void quadratic_loss(float* y, float* y_pred, float* res, int n) {
	for (int i = 0; i < n; ++i) {
		float diff = y[i] - y_pred[i];
		res[i] = diff*diff;
	}
}

void softmax(float* x, float* res, int classes, int n) {
	for (int col = 0; col < n; ++col) {
		int base = col*classes;
		float max = x[base];
		for (int i = 1; i < classes; ++i) {
			if (x[base+i] > max) max = x[base+i];
		}
		float sum = 0.f;
		for (int i = 0; i < classes; ++i) {
			res[base+i] = expf(x[base+i] - max);
			sum += res[base+i];
		}
		for (int i = 0; i < classes; ++i) {
			res[base+i] /= sum;
		}
	}
}

void softmax_loss(float* y, float* y_pred, float* res, int classes, int n) {
	for (int col = 0; col < n; ++col) {
		int base = col*classes;
		float sum = 0.f;
		for (int i = 0; i < classes; ++i) {
			float pred = y_pred[base+i];
			pred = clamp(pred, EPS, 1.f-EPS);
			sum += pred;
		}
		for (int i = 0; i < classes; ++i) {
			float pred = y_pred[base+i];
			pred = clamp(pred, EPS, 1.f-EPS) / sum;
			res[base+i] = -y[base+i] * logf(pred);
		}
	}
}

dnnError_t callCPU(Args* a) {
	dnnError_t error = 0;
	switch (a->op) {
	case COPY:
		memcpy(a->p[1], a->p[0], a->i[0]*4);
		break;
	case COPY_COL:
		memcpy(a->p[0] + a->i[0]*a->i[1]*4, a->p[1], a->i[1]*4);
		break;
	case COPY_ROW:
		array_copy_row(IP(a->p[0]), IP(a->p[1]), a->i[0], a->i[1], a->i[2]);
		break;
	case NEQ:
		array_neq(IP(a->p[0]), IP(a->p[1]), IP(a->p[2]), a->i[0]);
		break;
	case TILE0:
		array_tile0(a->p[0], a->p[1], a->i[0]*4, a->i[1]);
		break;
	case TILE1:
		array_tile1(FP(a->p[0]), FP(a->p[1]), a->i[0], a->i[1]);
		break;
	case FILL:
		if (a->i[0] == I32) {
			array_fill_i(IP(a->p[0]), a->f[0], a->i[1]);				
		} else {
			array_fill_f(FP(a->p[0]), a->f[0], a->i[1]);
		}
		break;
	case SUM:
		if (a->i[0] == I32) {
			array_sum_i(IP(a->p[0]), a->i[1], a->f[0], FP(a->p[1]));
		} else {
			array_sum_f(FP(a->p[0]), a->i[1], a->f[0], FP(a->p[1]));
		}
		break;
	case SCALE:
		cblas_sscal(a->i[0], a->f[0], FP(a->p[0]), 1);
		break;
	case AXPY:
		cblas_saxpy(a->i[0], a->f[0], FP(a->p[0]), 1, FP(a->p[1]), 1);
		break;
	case TRANS:
#ifdef USE_MKL
		mkl_somatcopy('c', 't', a->i[0], a->i[1], 1.0f, FP(a->p[0]), 
			a->i[0], FP(a->p[1]), a->i[1]);
#else
		cblas_somatcopy(CblasColMajor, CblasTrans, a->i[0], a->i[1], 1.0f, FP(a->p[0]), 
			a->i[0], FP(a->p[1]), a->i[1]);
#endif
		break;
	case GEMV:
		cblas_sgemv(CblasColMajor, a->i[0], a->i[1], a->i[2], a->f[0],
			FP(a->p[0]), a->i[1], FP(a->p[1]), 1, a->f[1], FP(a->p[2]), 1);
		break;
	case GEMM:
		cblas_sgemm(CblasColMajor, a->i[0], a->i[1], a->i[2], a->i[3], a->i[4], a->f[0], 
			FP(a->p[0]), a->i[5], FP(a->p[1]), a->i[6], a->f[1], FP(a->p[2]), a->i[7]);
		break;
	case SIGMOID:
		sigmoid_a(FP(a->p[0]), FP(a->p[1]), a->i[0]);
		break;
	case SIGMOID_D:
		sigmoid_d(FP(a->p[0]), FP(a->p[1]), FP(a->p[2]), a->i[0]);
		break;
	case TANH:
		tanh_a(FP(a->p[0]), FP(a->p[1]), a->i[0]);
		break;
	case TANH_D:
		tanh_d(FP(a->p[0]), FP(a->p[1]), FP(a->p[2]), a->i[0]);
		break;
	case RELU:
		relu_a(FP(a->p[0]), FP(a->p[1]), a->i[0]);
		break;
	case RELU_D:
		relu_d(FP(a->p[0]), FP(a->p[1]), FP(a->p[2]), a->i[0]);
		break;
	case ONEHOT:
		onehot(IP(a->p[0]), FP(a->p[1]), a->i[0], a->i[1]);
		break;
	case UNHOT:
		unhot(FP(a->p[0]), IP(a->p[1]), a->i[0], a->i[1]);
		break;
	case QUAD_LOSS:
		quadratic_loss(FP(a->p[0]), FP(a->p[1]), FP(a->p[2]), a->i[0]);
		break;
	case SOFTMAX:
		softmax(FP(a->p[0]), FP(a->p[1]), a->i[0], a->i[1]);
		break;
	case SOFTMAX_LOSS:
		softmax_loss(FP(a->p[0]), FP(a->p[1]), FP(a->p[2]), a->i[0], a->i[1]);
		break;;
	case DNN_EXECUTE:
#ifdef USE_MKL
		error = dnnExecute_F32((dnnPrimitive_t)(a->p[0]), a->p[1]);
#else
		error = E_UNIMPLEMENTED;
#endif
		break;
	}
	return error;
}

void set_num_threads(int n) {
#ifdef USE_MKL
	mkl_set_num_threads(n);
#else
	openblas_set_num_threads(n);
#endif
}

// call batch of commands
dnnError_t execCPU(int nargs, Args** buffer, int* ix) {
	for (int i = 0; i < nargs; ++i) {
		dnnError_t error = callCPU(buffer[i]);
		if (error != E_SUCCESS) {
			*ix = i;
			return error;
		}
	}
	return E_SUCCESS;
}

// call with profiling enabled
dnnError_t execCPUProfile(int nargs, Args** buffer, int* ix) {
	struct timeval start, end;
	for (int i = 0; i < nargs; ++i) {
		gettimeofday(&start, NULL);
		dnnError_t error = callCPU(buffer[i]);
		gettimeofday(&end, NULL);
		buffer[i]->usec = 1000000*(end.tv_sec-start.tv_sec) + end.tv_usec-start.tv_usec;
		if (error != E_SUCCESS) {
			*ix = i;
			return error;
		}
	}
	return E_SUCCESS;
}


