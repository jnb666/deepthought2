// Cuda kernel definitions
#define BLOCK 256
#define BLOCK2 16
#define EPS 1e-7f

#define clamp(x, min, max)  ((x) < (min) ? (min) : ((x) > (max) ? (max) : (x)))

__global__ void fill_f(float* a, float val, int n) {
	int i = blockIdx.x*blockDim.x + threadIdx.x;
	if (i < n) a[i] = val;
}

__global__ void fill_i(int* a, int val, int n) {
	int i = blockIdx.x*blockDim.x + threadIdx.x;
	if (i < n) a[i] = val;
}

__global__ void tile0(float* dst, const float* src, int rows, int cols) {
	int row = blockIdx.x*blockDim.x + threadIdx.x;
	int col = blockIdx.y*blockDim.y + threadIdx.y;
	if (row < rows && col < cols) dst[row+col*rows] = src[row];
}

__global__ void tile1(float* dst, const float* src, int rows, int cols) {
	int row = blockIdx.x*blockDim.x + threadIdx.x;
	int col = blockIdx.y*blockDim.y + threadIdx.y;
	if (row < rows && col < cols) dst[row+col*rows] = src[col];
}

__global__ void sum_f(float* in, float* total, int n) {
	int tid = threadIdx.x;
	int i = blockIdx.x*blockDim.x + threadIdx.x;
	__shared__ float x[BLOCK];
	x[tid] = (i < n) ? in[i] : 0.f;
	__syncthreads();
	for (int s = blockDim.x/2; s > 0; s /= 2) {
		if (tid < s) x[tid] += x[tid+s];
		__syncthreads();
	}
	if (tid == 0) {
		atomicAdd(total, x[tid]);
	}
}

__global__ void sum_i(int* in, float* total, int n) {
	int tid = threadIdx.x;
	int i = blockIdx.x*blockDim.x + threadIdx.x;
	__shared__ float x[BLOCK];
	x[tid] = (i < n) ? (float)(in[i]) : 0.f;
	__syncthreads();
	for (int s = blockDim.x/2; s > 0; s /= 2) {
		if (tid < s) x[tid] += x[tid+s];
		__syncthreads();
	}
	if (tid == 0) {
		atomicAdd(total, x[tid]);
	}
}

__global__ void neq(int* a, int* b, int* res, int n) {
	int i = blockIdx.x*blockDim.x + threadIdx.x;
	if (i < n) res[i] = (a[i] != b[i]);
}

__global__ void mul_elem(float* a, float* b, float* res, int n) {
	int i = blockIdx.x*blockDim.x + threadIdx.x;
	if (i < n) res[i] = a[i]*b[i];	
}

__global__ void onehot_1(int* y, float* y_one_hot, int n) {
	int i = blockIdx.x*blockDim.x + threadIdx.x;
	if (i < n) y_one_hot[i] = (float)(y[i]);
}

__global__ void onehot(int* y, float* y_one_hot, int n, int classes) {
	int i = blockIdx.x*blockDim.x + threadIdx.x;
	if (i < n) y_one_hot[y[i]+i*classes] = 1.f;
}

__global__ void unhot_1(float* y_one_hot, int* y, int n) {
	int i = blockIdx.x*blockDim.x + threadIdx.x;
	if (i < n) y[i] = y_one_hot[i] > 0.5;
}

__global__ void unhot(float* y_one_hot, int* y, int n, int classes) {
	int i = blockIdx.x*blockDim.x + threadIdx.x;
	if (i >= n) return;
	int ix = 0;
	int base = i*classes;
	float max = y_one_hot[base];
	for (int j = 1; j < classes; ++j) {
		if (y_one_hot[base+j] > max) {
			max = y_one_hot[base+j];
			ix = j;
		}
	}
	y[i] = ix;
}

__global__ void quadratic_loss(float* y, float* y_pred, float* res, int n) {
	int i = blockIdx.x*blockDim.x + threadIdx.x;
	if (i < n) {
		float diff = y[i] - y_pred[i];
		res[i] = diff*diff;
	}
}

__global__ void softmax(float* x, float* res, int classes, int n) {
	int i = blockIdx.x*blockDim.x + threadIdx.x;
	if (i >= n) return;
	int base = i*classes;
	float max = x[base];
	for (int j = 1; j < classes; ++j) {
		if (x[base+j] > max) max = x[base+j];
	}
	float sum = 0.f;
	for (int j = 0; j < classes; ++j) {
		res[base+j] = expf(x[base+j] - max);
		sum += res[base+j];
	}
	for (int j = 0; j < classes; ++j) {
		res[base+j] /= sum;
	}
}

__global__ void softmax_loss(float* y, float* y_pred, float* res, int classes, int n) {
	int i = blockIdx.x*blockDim.x + threadIdx.x;
	if (i >= n) return;
	int base = i*classes;
	float sum = 0.f;
	for (int j = 0; j < classes; ++j) {
		float pred = y_pred[base+j];
		pred = clamp(pred, EPS, 1.f-EPS);
		sum += pred;
	}
	for (int j = 0; j < classes; ++j) {
		float pred = y_pred[base+j];
		pred = clamp(pred, EPS, 1.f-EPS) / sum;
		res[base+j] = -y[base+j] * logf(pred);
	}
}

extern "C" { 

void cuda_fill_f(cudaStream_t stream, float* a, float val, int n) {
	fill_f<<<(n+BLOCK-1)/BLOCK, BLOCK, 0, stream>>>(a, val, n);
}

void cuda_fill_i(cudaStream_t stream, int* a, float val, int n) {
	fill_i<<<(n+BLOCK-1)/BLOCK, BLOCK, 0, stream>>>(a, (int)val, n);
}

// tile column vector
void cuda_tile0(cudaStream_t stream, float* dst, float* src, int rows, int cols) {
	dim3 numBlocks((rows+BLOCK2-1)/BLOCK2, (cols+BLOCK2-1)/BLOCK2);
	dim3 threadsPerBlock(BLOCK2, BLOCK2);
	tile0<<<numBlocks, threadsPerBlock, 0, stream>>>(dst, src, rows, cols);
}

// tile row vector
void cuda_tile1(cudaStream_t stream, float* dst, float* src, int rows, int cols) {
	dim3 numBlocks((rows+BLOCK2-1)/BLOCK2, (cols+BLOCK2-1)/BLOCK2);
	dim3 threadsPerBlock(BLOCK2, BLOCK2);
	tile1<<<numBlocks, threadsPerBlock, 0, stream>>>(dst, src, rows, cols);
}

void cuda_sum_f(cudaStream_t stream, float* in, float* total, int n) {
	if (cudaMemset(total, 0, 4) == cudaSuccess) {
		sum_f<<<(n+BLOCK-1)/BLOCK, BLOCK>>>(in, total, n);
	}
}

void cuda_sum_i(cudaStream_t stream, int* in, float* total, int n) {
	if (cudaMemset(total, 0, 4) == cudaSuccess) {
		sum_i<<<(n+BLOCK-1)/BLOCK, BLOCK>>>(in, total, n);
	}
}

void cuda_neq(cudaStream_t stream, int* a, int* b, int* res, int n) {
	neq<<<(n+BLOCK-1)/BLOCK, BLOCK>>>(a, b, res, n);
}

void cuda_mul_elem(cudaStream_t stream, float* a, float* b, float* res, int n) {
	mul_elem<<<(n+BLOCK-1)/BLOCK, BLOCK>>>(a, b, res, n);
}

void cuda_onehot(cudaStream_t stream, int* y, float* y_one_hot, int n, int classes) {
	if (classes == 1) {
		onehot_1<<<(n+BLOCK-1)/BLOCK, BLOCK>>>(y, y_one_hot, n);
	} else {
		if (cudaMemset(y_one_hot, 0, n*classes*4) == cudaSuccess) {
			onehot<<<(n+BLOCK-1)/BLOCK, BLOCK>>>(y, y_one_hot, n, classes);
		}
	}
}

void cuda_unhot(cudaStream_t stream, float* y_one_hot, int* y, int n, int classes) {
	if (classes == 1) {
		unhot_1<<<(n+BLOCK-1)/BLOCK, BLOCK>>>(y_one_hot, y, n);
	} else {
		unhot<<<(n+BLOCK-1)/BLOCK, BLOCK>>>(y_one_hot, y, n, classes);
	}
}

void cuda_quad_loss(cudaStream_t stream, float* y, float* y_pred, float* res, int n) {
	quadratic_loss<<<(n+BLOCK-1)/BLOCK, BLOCK>>>(y, y_pred, res, n);
}

void cuda_softmax(cudaStream_t stream, float* x, float* res, int classes, int n) {
	softmax<<<(n+BLOCK-1)/BLOCK, BLOCK>>>(x, res, classes, n);
}

void cuda_softmax_loss(cudaStream_t stream, float* y, float* y_pred, float* res, int classes, int n) {
	softmax_loss<<<(n+BLOCK-1)/BLOCK, BLOCK>>>(y, y_pred, res, classes, n);
}

}