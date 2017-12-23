// Package cuda contains wrapper functions for Cuda api
package cuda

/*
#cgo CFLAGS: -I /usr/local/cuda/include
#cgo LDFLAGS: -L /usr/local/cuda/lib64 -lcublas -lcudnn -lcudart
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <cudnn.h>
*/
import "C"

import (
	"errors"
	"fmt"
	"runtime"
	"unsafe"
)

type Device C.int

// Get new device
func NewDevice() Device {
	var deviceCount C.int
	chk(C.cudaGetDeviceCount(&deviceCount))
	if deviceCount < 1 {
		panic("No Cuda device found")
	}
	// todo: handle multiple devices, for now just return the first one
	dev := C.int(0)
	chk(C.cudaSetDevice(dev))
	return Device(dev)
}

type Stream struct {
	stream C.cudaStream_t
	blas   C.cublasHandle_t
	cudnn  C.cudnnHandle_t
}

// Allocate new Cuda stream and associate cuBlas and cuDNN context with this.
func NewStream() *Stream {
	s := new(Stream)
	chk(C.cudaStreamCreate(&s.stream))
	chkBlas(C.cublasCreate(&s.blas))
	chkBlas(C.cublasSetStream(s.blas, s.stream))
	chkDnn(C.cudnnCreate(&s.cudnn))
	chkDnn(C.cudnnSetStream(s.cudnn, s.stream))
	runtime.SetFinalizer(s, func(obj *Stream) { obj.Release() })
	return s
}

func (s *Stream) Sync() {
	chk(C.cudaStreamSynchronize(s.stream))
}

func (s *Stream) Release() {
	C.cudnnDestroy(s.cudnn)
	C.cublasDestroy(s.blas)
	C.cudaStreamDestroy(s.stream)
}

type Buffer struct {
	Ptr  unsafe.Pointer
	Size int
}

// Allocate a buffer on the GPU with given number of bytes
func NewBuffer(size int) Buffer {
	buf := Buffer{Size: size}
	if size > 0 {
		chk(C.cudaMalloc(&buf.Ptr, C.size_t(size)))
	}
	return buf
}

func (b Buffer) Clear() Buffer {
	chk(C.cudaMemset(b.Ptr, 0, C.size_t(b.Size)))
	return b
}

func (b Buffer) Free() {
	if b.Size > 0 {
		C.cudaFree(b.Ptr)
		b.Size = 0
	}
}

type Error C.cudaError_t

// Convert Cuda error code to go error
func GetError(err Error) error {
	return getError(C.cudaError_t(err))
}

// Check for error running Cuda function, panics if not success
func chk(err C.cudaError_t) {
	if e := getError(err); e != nil {
		panic(e)
	}
}

func getError(err C.cudaError_t) error {
	if err == C.cudaSuccess {
		return nil
	}
	cstr := C.cudaGetErrorString(err)
	return fmt.Errorf("Cuda error: %s", C.GoString(cstr))
}

type BlasStatus C.cublasStatus_t

func chkBlas(err C.cublasStatus_t) {
	if e := GetBlasError(BlasStatus(err)); e != nil {
		panic(e)
	}
}

// Convert cuBlas status code to go error
func GetBlasError(err BlasStatus) error {
	switch err {
	case C.CUBLAS_STATUS_SUCCESS:
		return nil
	case C.CUBLAS_STATUS_NOT_INITIALIZED:
		return errors.New("clBlas: not initialized")
	case C.CUBLAS_STATUS_ALLOC_FAILED:
		return errors.New("clBlas: allocation failed")
	case C.CUBLAS_STATUS_INVALID_VALUE:
		return errors.New("clBlas: invalid value")
	case C.CUBLAS_STATUS_ARCH_MISMATCH:
		return errors.New("clBlas: architecture mismatch")
	case C.CUBLAS_STATUS_MAPPING_ERROR:
		return errors.New("clBlas: texture mapping failed")
	case C.CUBLAS_STATUS_EXECUTION_FAILED:
		return errors.New("clBlas: execution failed")
	case C.CUBLAS_STATUS_INTERNAL_ERROR:
		return errors.New("clBlas: internal error")
	case C.CUBLAS_STATUS_NOT_SUPPORTED:
		return errors.New("clBlas: not supported")
	case C.CUBLAS_STATUS_LICENSE_ERROR:
		return errors.New("clBlas: license error")
	default:
		return errors.New("clBlas: unknown error!")
	}
}
