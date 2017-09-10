// Package num contains numeric Array processing routines such as optimised matix multiplication.
package num

/*
#cgo CFLAGS: -g -O2 -std=c99 -I/opt/OpenBLAS/include
#cgo LDFLAGS: -L/opt/OpenBLAS/lib -lopenblas -lm -lpthread
#include "num.h"
*/
import "C"

import (
	"fmt"
	"reflect"
	"unsafe"
)

// Data type of an element of the array
type DataType int

const (
	Int32   DataType = C.I32
	Float32 DataType = C.F32
)

// TransType flag indicates if matrix is transposed
type TransType int

const (
	NoTrans TransType = C.CblasNoTrans
	Trans   TransType = C.CblasTrans
)

// Read data from array into a slice.
func Read(a Array, data interface{}) Function {
	return args(C.COPY, Prod(a.Dims()), ptr(data), a.Data())
}

// Write data from a slice into the given array.
func Write(a Array, data interface{}) Function {
	return args(C.COPY, Prod(a.Dims()), a.Data(), ptr(data))
}

// Write to one row in the array
func WriteRow(a Array, row int, data interface{}) Function {
	dims := a.Dims()
	if len(dims) > 2 || len(dims) < 1 {
		panic("WriteRow: must be vector or matrix")
	}
	cols := 1
	if len(dims) == 2 {
		cols = a.Dims()[1]
	}
	return args(C.COPY_ROW, row, a.Dims()[0], cols, a.Data(), ptr(data))
}

// Fill array with a scalar value
func Fill(a Array, scalar float32) Function {
	return args(C.FILL, int(a.Dtype()), Prod(a.Dims()), scalar, a.Data())
}

// Copy from src to dst, broadcast vector to matrix if needed
func Copy(dst, src Array) Function {
	if src.Dtype() != dst.Dtype() {
		panic("Copy: arguments must be same type")
	}
	ddim, sdim := dst.Dims(), src.Dims()
	if SameShape(ddim, sdim) {
		return args(C.COPY, Prod(ddim), dst.Data(), src.Data())
	} else if len(sdim) == 1 && len(ddim) == 2 && sdim[0] == ddim[0] {
		return args(C.TILE0, sdim[0], ddim[1], dst.Data(), src.Data())
	} else if len(sdim) == 1 && len(ddim) == 2 && sdim[0] == ddim[1] {
		return args(C.TILE1, sdim[0], ddim[0], dst.Data(), src.Data())
	} else {
		panic(fmt.Sprintf("Copy: cannot copy from %v to %v shape", sdim, ddim))
	}
}

// Element wise != comparison
func Neq(x, y, res Array) Function {
	if x.Dtype() != Int32 || y.Dtype() != Int32 || res.Dtype() != Int32 {
		panic("Neq: incorrect datatype")
	}
	if !SameShape(x.Dims(), res.Dims()) || !SameShape(y.Dims(), res.Dims()) {
		panic("Neq: arrays must be same shape")
	}
	n := Prod(x.Dims())
	return args(C.NEQ, n, x.Data(), y.Data(), res.Data())
}

// Convert to one hot representation
func Onehot(x, y Array, classes int) Function {
	if x.Dtype() != Int32 || y.Dtype() != Float32 {
		panic("Onehot: incorrect datatype")
	}
	xdim, ydim := x.Dims(), y.Dims()
	if len(xdim) != 1 || len(ydim) != 2 || xdim[0] != ydim[0] || ydim[1] != classes {
		panic("Onehot: invalid array shape")
	}
	return args(C.ONEHOT, xdim[0], classes, x.Data(), y.Data())
}

// Convert from OneHot format back to labels
func Unhot(x, y Array) Function {
	if x.Dtype() != Float32 || y.Dtype() != Int32 {
		panic("Unhot: incorrect datatype")
	}
	xdim, ydim := x.Dims(), y.Dims()
	if len(xdim) != 2 || len(ydim) != 1 || xdim[0] != ydim[0] {
		panic("Unhot: invalid array shape")
	}
	return args(C.UNHOT, xdim[0], xdim[1], x.Data(), y.Data())
}

// Scale array: x <- alpha*x
func Scale(alpha float32, x Array) Function {
	if x.Dtype() != Float32 {
		panic("Axpy: dtype must by Float32")
	}
	return args(C.SCALE, Prod(x.Dims()), alpha, x.Data())
}

// Array addition and scaling: y <- alpha*x + y
func Axpy(alpha float32, x, y Array) Function {
	if x.Dtype() != Float32 || y.Dtype() != Float32 {
		panic("Axpy: dtype must by Float32")
	}
	if !SameShape(x.Dims(), y.Dims()) {
		panic("Axpy: arrays must be same shape")
	}
	n := Prod(x.Dims())
	return args(C.AXPY, n, alpha, x.Data(), y.Data())
}

// Transpose sets mB to a copy of mA with the data transposed.
func Transpose(mA, mB Array) Function {
	adim, bdim := mA.Dims(), mB.Dims()
	if len(adim) != 2 || len(bdim) != 2 {
		panic("Transpose: arrays must be 2D")
	}
	if adim[0] != bdim[1] || adim[1] != bdim[0] {
		panic("Transpose: destination matrix is wrong shape")
	}
	return args(C.TRANS, adim[0], adim[1], mA.Data(), mB.Data())
}

// Calculate the scalar sum of the values in the array. Multiplies each result by scale.
func Sum(a, total Array, scale float32) Function {
	if len(total.Dims()) != 0 || total.Dtype() != Float32 {
		panic("Sum: result type should be float32 scalar")
	}
	return args(C.SUM, int(a.Dtype()), Prod(a.Dims()), scale, a.Data(), total.Data())
}

// Matrix vector multiplication: y <- alpha*dot(mA,x) + beta*y
func Gemv(alpha, beta float32, mA, x, y Array, aTrans TransType) Function {
	if mA.Dtype() != Float32 || x.Dtype() != Float32 || y.Dtype() != Float32 {
		panic("Gemv: dtype must by Float32")
	}
	adim, xdim, ydim := mA.Dims(), x.Dims(), y.Dims()
	if len(adim) != 2 || len(xdim) != 1 || len(ydim) != 1 {
		panic("Gemv: must have matrix and vector inputs")
	}
	m, n := adim[0], adim[1]
	if aTrans == Trans {
		if xdim[0] != m || ydim[0] != n {
			panic("Gemv: incorrect vector size")
		}
	} else {
		if xdim[0] != n || ydim[0] != m {
			panic("Gemv: incorrect vector size")
		}
	}
	return args(C.GEMV, int(aTrans), m, n, alpha, beta, mA.Data(), x.Data(), y.Data())
}

// Matrix matrix multiplication: mC <- alpha*dot(mA, mB) + beta*mC
func Gemm(alpha, beta float32, mA, mB, mC Array, aTrans, bTrans TransType) Function {
	if mA.Dtype() != Float32 || mB.Dtype() != Float32 || mC.Dtype() != Float32 {
		panic("Gemm: dtype must by Float32")
	}
	adim, bdim, cdim := mA.Dims(), mB.Dims(), mC.Dims()
	if len(adim) != 2 || len(bdim) != 2 || len(cdim) != 2 {
		panic("Gemm: must have 2 dimensional arrays")
	}
	m, k := adim[0], adim[1]
	k2, n := bdim[0], bdim[1]
	if aTrans == Trans {
		m, k = k, m
	}
	if bTrans == Trans {
		k2, n = n, k2
	}
	if k2 != k {
		panic(fmt.Sprintf("Gemm: invalid input shape %v x %v", adim, bdim))
	}
	if cdim[0] != m || cdim[1] != n {
		panic(fmt.Sprintf("Gemm: invalid output shape %v expecting [%d %d]", cdim, m, n))
	}
	return args(C.GEMM, int(aTrans), int(bTrans), m, n, k, adim[0], bdim[0], cdim[0],
		alpha, beta, mA.Data(), mB.Data(), mC.Data())
}

// Sigmoid activation function: y = 1/(1+e**(-x))
func Sigmoid(x, y Array) Function {
	return unaryFunc(C.SIGMOID, x, y)
}

func SigmoidD(x, grad, y Array) Function {
	return binaryFunc(C.SIGMOID_D, x, grad, y)
}

// Tanh activation function: y = tanh(x)
func Tanh(x, y Array) Function {
	return unaryFunc(C.TANH, x, y)
}

func TanhD(x, grad, y Array) Function {
	return binaryFunc(C.TANH_D, x, grad, y)
}

// Relu rectified linear activation function: y = max(x, 0)
func Relu(x, y Array) Function {
	return unaryFunc(C.RELU, x, y)
}

func ReluD(x, grad, y Array) Function {
	return binaryFunc(C.RELU_D, x, grad, y)
}

// Quadratic loss function: (x-y)**2
func QuadraticLoss(x, y, res Array) Function {
	return binaryFunc(C.QUAD_LOSS, x, y, res)
}

// Softmax activation function
func Softmax(x, res Array) Function {
	if x.Dtype() != Float32 || res.Dtype() != Float32 {
		panic("Softmax: dtype must by Float32")
	}
	xdim, rdim := x.Dims(), res.Dims()
	if len(xdim) != 2 || !SameShape(xdim, rdim) {
		panic("Softmax: arrays must be 2d and same shape")
	}
	return args(C.SOFTMAX, xdim[0], xdim[1], x.Data(), res.Data())
}

// Softmax loss function
func SoftmaxLoss(x, y, res Array) Function {
	if x.Dtype() != Float32 || y.Dtype() != Float32 || res.Dtype() != Float32 {
		panic("Softmax: dtype must by Float32")
	}
	xdim, ydim, rdim := x.Dims(), y.Dims(), res.Dims()
	if len(xdim) != 2 || !SameShape(xdim, ydim) || !SameShape(xdim, rdim) {
		panic("Softmax: arrays must be 2d and same shape")
	}
	return args(C.SOFTMAX_LOSS, xdim[0], xdim[1], x.Data(), y.Data(), res.Data())
}

func unaryFunc(op int, x, y Array) Function {
	if x.Dtype() != Float32 || y.Dtype() != Float32 {
		panic("UnaryFunc: dtype must by Float32")
	}
	if !SameShape(x.Dims(), y.Dims()) {
		panic("UnaryFunc: arrays must be same shape")
	}
	return args(op, Prod(x.Dims()), x.Data(), y.Data())
}

func binaryFunc(op int, x, y, z Array) Function {
	if x.Dtype() != Float32 || y.Dtype() != Float32 || z.Dtype() != Float32 {
		panic("BinaryFunc: dtype must by Float32")
	}
	if !SameShape(x.Dims(), z.Dims()) || !SameShape(y.Dims(), z.Dims()) {
		panic("BinaryFunc: arrays must be same shape")
	}
	return args(op, Prod(x.Dims()), x.Data(), y.Data(), z.Data())
}

// Function which may be called via the queue
type Function *C.struct_args

func args(op int, arg ...interface{}) Function {
	a := &C.struct_args{op: C.int(op)}
	ni, nf, np := 0, 0, 0
	for _, val := range arg {
		switch v := val.(type) {
		case int:
			a.i[ni] = C.int(v)
			ni++
		case float32:
			a.f[nf] = C.float(v)
			nf++
		case unsafe.Pointer:
			a.p[np] = v
			np++
		default:
			panic(fmt.Sprintf("invalid arg type: %T", val))
		}
	}
	return a
}

func setCPUThreads(threads int) {
	if threads < 1 {
		threads = 1
	}
	fmt.Println("set num threads", threads)
	C.openblas_set_num_threads(C.int(threads))
}

func execCPU(buffer []Function) {
	ptr := unsafe.Pointer(&buffer[0])
	C.execCPU(C.int(len(buffer)), (**C.struct_args)(ptr))
}

func ptr(ival interface{}) unsafe.Pointer {
	v := reflect.ValueOf(ival)
	return unsafe.Pointer(v.Pointer())
}
