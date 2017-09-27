// Cgo interface routines for cuda functions
#include "num.h"
#include "cuda.h"

#define CU_TRANS(x) (((x) == CblasTrans) ? CUBLAS_OP_T : CUBLAS_OP_N)

void callGPU(Args* a, Stream* s, cublasStatus_t* blasStatus, cudnnStatus_t* dnnStatus) {
	float zero = 0.f;
	float one = 1.f;
	switch (a->op) {
	case COPY:
		cudaMemcpyAsync(a->p[1], a->p[0], a->i[0]*4, cudaMemcpyDeviceToDevice, s->stream);
		break;
	case COPY_TO_DEVICE:
		cudaMemcpyAsync(a->p[1], a->p[0], a->i[0]*4, cudaMemcpyHostToDevice, s->stream);
		break;
	case COPY_TO_HOST:
		cudaMemcpyAsync(a->p[1], a->p[0], a->i[0]*4, cudaMemcpyDeviceToHost, s->stream);
		break;
	case COPY_COL:
		cudaMemcpyAsync(a->p[0] + a->i[0]*a->i[1]*4, a->p[1], a->i[1]*4, cudaMemcpyHostToDevice, s->stream);
		break;
	case TILE0:
		cuda_tile0(s->stream, a->p[0], a->p[1], a->i[0], a->i[1]);
		break;
	case TILE1:
		cuda_tile1(s->stream, a->p[0], a->p[1], a->i[0], a->i[1]);
		break;
	case FILL:
		if (a->i[0] == I32) {
			cuda_fill_i(s->stream, IP(a->p[0]), a->f[0], a->i[1]);				
		} else {
			cuda_fill_f(s->stream, FP(a->p[0]), a->f[0], a->i[1]);
		}
		break;
	case NEQ:
		cuda_neq(s->stream, IP(a->p[0]), IP(a->p[1]), IP(a->p[2]), a->i[0]);
		break;
	case SUM:
		if (a->i[0] == I32) {
			cuda_sum_i(s->stream, IP(a->p[0]), FP(a->p[1]), a->i[1]);
		} else {
			cuda_sum_f(s->stream, FP(a->p[0]), FP(a->p[1]), a->i[1]);
		}
		break;
	case AXPY:
		*blasStatus = cublasSaxpy(s->blas, a->i[0], &(a->f[0]), FP(a->p[0]), 1, FP(a->p[1]), 1); 
		break;
	case TRANS:
		*blasStatus = cublasSgeam(s->blas, CUBLAS_OP_T, CUBLAS_OP_N, a->i[1], a->i[0], 
			&one, FP(a->p[0]), a->i[0], &zero, FP(a->p[1]), a->i[1], FP(a->p[1]), a->i[1]);
		break;
	case GEMV:
		*blasStatus = cublasSgemv(s->blas, CU_TRANS(a->i[0]), a->i[1], a->i[2], &one,
			FP(a->p[0]), a->i[1], FP(a->p[1]), 1, &zero, FP(a->p[2]), 1);
		break;
	case GEMM:
		*blasStatus = cublasSgemm(s->blas, CU_TRANS(a->i[0]), CU_TRANS(a->i[1]), a->i[2], a->i[3], a->i[4], &one, 
			FP(a->p[0]), a->i[5], FP(a->p[1]), a->i[6], &(a->f[0]), FP(a->p[2]), a->i[7]);
		break;
	case ONEHOT:
		cuda_onehot(s->stream, IP(a->p[0]), FP(a->p[1]), a->i[0], a->i[1]);
		break;
	case UNHOT:
		cuda_unhot(s->stream, FP(a->p[0]), IP(a->p[1]), a->i[0], a->i[1]);
		break;
	case QUAD_LOSS:
		cuda_quad_loss(s->stream, FP(a->p[0]), FP(a->p[1]), FP(a->p[2]), a->i[0]);
		break;
	case SOFTMAX:
		cuda_softmax(s->stream, FP(a->p[0]), FP(a->p[1]), a->i[0], a->i[1]);
		break;
	case SOFTMAX_LOSS:
		cuda_softmax_loss(s->stream, FP(a->p[0]), FP(a->p[1]), FP(a->p[2]), a->i[0], a->i[1]);
		break;
	case CUDNN_ACTIV_FPROP:
		*dnnStatus = cudnnActivationForward(s->cudnn, (cudnnActivationDescriptor_t)(a->p[0]), 
			&one, (cudnnTensorDescriptor_t)(a->p[1]), a->p[2], 			// src
			&zero, (cudnnTensorDescriptor_t)(a->p[1]), a->p[3]);		// dst
		break;
	case CUDNN_ACTIV_BPROP:
		*dnnStatus = cudnnActivationBackward(s->cudnn, (cudnnActivationDescriptor_t)(a->p[0]), 
			&one, (cudnnTensorDescriptor_t)(a->p[1]), a->p[2],			// dst
			 (cudnnTensorDescriptor_t)(a->p[1]), a->p[3],				// diffDst
			 (cudnnTensorDescriptor_t)(a->p[1]), a->p[4],				// prev src
			&zero, (cudnnTensorDescriptor_t)(a->p[1]), a->p[5]);		// diffSrc
		break;
	case CUDNN_CONV_FPROP:
		*dnnStatus = cudnnConvolutionForward(s->cudnn,
			&one, (cudnnTensorDescriptor_t)(a->p[3]), a->p[6], 			// src
			(cudnnFilterDescriptor_t)(a->p[2]), a->p[5], 				// filter
			(cudnnConvolutionDescriptor_t)(a->p[0]), 
			(cudnnConvolutionFwdAlgo_t)(a->i[0]),
			a->p[1], (size_t)(a->i[1]), 								// workspace
			&zero, (cudnnTensorDescriptor_t)(a->p[4]), a->p[7]);		// dst
		break;
	case CUDNN_CONV_FPROP_BIAS:
		*dnnStatus = cudnnAddTensor(s->cudnn, 
			&one, (cudnnTensorDescriptor_t)(a->p[0]), a->p[2],			// bias 
			&one, (cudnnTensorDescriptor_t)(a->p[1]), a->p[3]);			// dst
		break;
	case CUDNN_CONV_BPROP_DATA:
		*dnnStatus = cudnnConvolutionBackwardData(s->cudnn,
			&one, (cudnnFilterDescriptor_t)(a->p[2]), a->p[5], 			// filter
			(cudnnTensorDescriptor_t)(a->p[3]), a->p[6], 				// diffDst
			(cudnnConvolutionDescriptor_t)(a->p[0]), 
			(cudnnConvolutionBwdDataAlgo_t)(a->i[0]),
			a->p[1], (size_t)(a->i[1]),									// workspace
			&zero, (cudnnTensorDescriptor_t)(a->p[4]), a->p[7]);		// diffSrc
		break;
	case CUDNN_CONV_BPROP_FILTER:
		*dnnStatus = cudnnConvolutionBackwardFilter(s->cudnn,
			&one, (cudnnTensorDescriptor_t)(a->p[2]), a->p[5],			// src
			(cudnnTensorDescriptor_t)(a->p[3]), a->p[6], 				// diffDst
			(cudnnConvolutionDescriptor_t)(a->p[0]), 
			(cudnnConvolutionBwdFilterAlgo_t)(a->i[0]),
			a->p[1], (size_t)(a->i[1]),									// workspace
			&zero, (cudnnFilterDescriptor_t)(a->p[4]), a->p[7]);		// dFilter
		break;
	case CUDNN_CONV_BPROP_BIAS:
		*dnnStatus = cudnnConvolutionBackwardBias(s->cudnn,
			&one, (cudnnTensorDescriptor_t)(a->p[0]), a->p[2], 			// diffDst
			&zero, (cudnnTensorDescriptor_t)(a->p[1]), a->p[3]);		// dBias
		break;
	case CUDNN_POOL_FPROP:
		*dnnStatus = cudnnPoolingForward(s->cudnn, (cudnnPoolingDescriptor_t)(a->p[0]), 
			&one, (cudnnTensorDescriptor_t)(a->p[1]), a->p[3], 			// src
			&zero, (cudnnTensorDescriptor_t)(a->p[2]), a->p[4]);		// dst
		break;
	case CUDNN_POOL_BPROP:
		*dnnStatus = cudnnPoolingBackward(s->cudnn, (cudnnPoolingDescriptor_t)(a->p[0]), 
			&one, (cudnnTensorDescriptor_t)(a->p[2]), a->p[3],			// dst
			(cudnnTensorDescriptor_t)(a->p[2]), a->p[4],				// diffDst
			(cudnnTensorDescriptor_t)(a->p[1]), a->p[5],				// prev src
			&zero, (cudnnTensorDescriptor_t)(a->p[1]), a->p[6]);		// diffSrc
		break;
	default:
		*blasStatus = CUBLAS_STATUS_NOT_SUPPORTED;
	}
}

void initEvents(Events* e) {
	for (int i = 0; i < QUEUE_SIZE; ++i) {
		cudaEventCreate(&(e->start[i]));
		cudaEventCreate(&(e->end[i]));
	}
}

// returns -1 if ok, -2 if Cuda error or index of call if cuBlas error
int execGPU(int nargs, Args** buffer, Stream* stream, cudaError_t* error, cublasStatus_t* blasError, cudnnStatus_t* dnnError) {
	*blasError = CUBLAS_STATUS_SUCCESS;
	*dnnError = CUDNN_STATUS_SUCCESS;
	for (int i = 0; i < nargs; ++i) {
		callGPU(buffer[i], stream, blasError, dnnError);
		if (*blasError != CUBLAS_STATUS_SUCCESS || *dnnError != CUDNN_STATUS_SUCCESS) {
			return i;
		}
	}
	cudaError_t status = cudaPeekAtLastError();
	if (status != cudaSuccess) {
		*error = status;
		return -2;
	}
	return -1;
}

// returns -1 if ok, -2 if Cuda error or index of call if cuBlas error
int execGPUProfile(int nargs, Args** buffer, Stream* stream, cudaError_t* error, cublasStatus_t* blasError, cudnnStatus_t* dnnError, Events* e) {
	*blasError = CUBLAS_STATUS_SUCCESS;
	*dnnError = CUDNN_STATUS_SUCCESS;
	for (int i = 0; i < nargs; ++i) {
		cudaEventRecord(e->start[i], stream->stream);
		callGPU(buffer[i], stream, blasError, dnnError);
		cudaEventRecord(e->end[i], stream->stream);
		if (*blasError != CUBLAS_STATUS_SUCCESS || *dnnError != CUDNN_STATUS_SUCCESS) {
			return i;
		}
	}
	cudaError_t status = cudaPeekAtLastError();
	if (status != cudaSuccess) {
		*error = status;
		return -2;
	}
	cudaEventSynchronize(e->end[nargs-1]);
	for (int i = 0; i < nargs; ++i) {
		cudaEventElapsedTime(&(buffer[i]->msec), e->start[i], e->end[i]);
	}
	return -1;
}
