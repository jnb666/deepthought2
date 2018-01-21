// Package num contains numeric Array processing routines such as optimised matix multiplication.
package num

/*
#cgo CFLAGS: -g -O2 -std=c99 -I/opt/intel/mkl/include -I/usr/local/cuda/include
#cgo LDFLAGS: -L. -l kernels -L/usr/local/cuda/lib64 -lcublas -lcudnn -lcudart
#cgo LDFLAGS: -L/opt/intel/mkl/lib/intel64 -L/opt/intel/tbb/lib/intel64/gcc4.7 -Wl,--no-as-needed -lmkl_intel_lp64 -lmkl_tbb_thread -lmkl_core -ltbb -lstdc++ -lpthread -lm -ldl
#include "num.h"
*/
import "C"

import (
	"fmt"
	"github.com/jnb666/deepthought2/num/cuda"
	"reflect"
	"unsafe"
)

var opName = map[C.int]string{
	C.COPY:            "copy",
	C.COPY_TO_DEVICE:  "copy_to_device",
	C.COPY_TO_HOST:    "copy_to_host",
	C.COPY_COL:        "copy_col",
	C.TILE0:           "tile0",
	C.TILE1:           "tile1",
	C.FILL:            "fill",
	C.NEQ:             "neq",
	C.ONEHOT:          "onehot",
	C.UNHOT:           "unhot",
	C.SCALE:           "scale",
	C.AXPY:            "axpy",
	C.TRANS:           "trans",
	C.SUM:             "sum",
	C.GEMV:            "gemv",
	C.GEMM:            "gemm",
	C.MUL_ELEM:        "mul_elem",
	C.SIGMOID:         "sigmoid",
	C.SIGMOID_D:       "sigmoid_d",
	C.TANH:            "tanh",
	C.TANH_D:          "tanh_d",
	C.RELU:            "relu",
	C.RELU_D:          "relu_d",
	C.QUAD_LOSS:       "quad_loss",
	C.SOFTMAX:         "sofmax",
	C.SOFTMAX_LOSS:    "softmax_loss",
	C.MKL_DNN_EXECUTE: "mkl_dnn",
}

func getOpName(op int) string {
	if op < C.CUDNN_EXECUTE {
		return opName[C.int(op)]
	}
	return cuda.OpName(op - C.CUDNN_EXECUTE)
}

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
	return args(C.COPY_TO_HOST, Prod(a.Dims()), a.Data(), ptr(data))
}

// Write data from a slice into the given array.
func Write(a Array, data interface{}) Function {
	return args(C.COPY_TO_DEVICE, Prod(a.Dims()), ptr(data), a.Data())
}

// Write to one column in the array
func WriteCol(a Array, col int, data interface{}) Function {
	dims := a.Dims()
	var rows, cols int
	if len(dims) == 1 {
		rows, cols = 1, dims[0]
	} else if len(dims) == 2 {
		rows, cols = dims[0], dims[1]
	} else {
		panic("WriteCol: must be vector or matrix")
	}
	if col < 0 || col >= cols {
		panic("WriteCol: column out of range")
	}
	return args(C.COPY_COL, col, rows, a.Data(), ptr(data))
}

// Fill array with a scalar value
func Fill(a Array, scalar float32) Function {
	return args(C.FILL, int(a.Dtype()), Prod(a.Dims()), scalar, a.Data())
}

// Copy from src to dst, broadcast vector to matrix if needed, vector is tiled row wise
func Copy(src, dst Array) Function {
	if src.Dtype() != dst.Dtype() {
		panic("Copy: arguments must be same type")
	}
	ddim, sdim := dst.Dims(), src.Dims()
	if SameShape(ddim, sdim) {
		return args(C.COPY, Prod(ddim), src.Data(), dst.Data())
	} else if len(sdim) == 1 && len(ddim) == 2 && sdim[0] == ddim[1] {
		return args(C.TILE1, ddim[0], ddim[1], dst.Data(), src.Data())
	} else if len(sdim) == 2 && sdim[1] == 1 && len(ddim) == 2 && sdim[0] == ddim[0] {
		return args(C.TILE0, ddim[0], ddim[1], dst.Data(), src.Data())
	} else if len(sdim) == 2 && sdim[0] == 1 && len(ddim) == 2 && sdim[1] == ddim[1] {
		return args(C.TILE1, ddim[0], ddim[1], dst.Data(), src.Data())
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
	if len(xdim) != 1 || len(ydim) != 2 || xdim[0] != ydim[1] || ydim[0] != classes {
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
	if len(xdim) != 2 || len(ydim) != 1 || xdim[1] != ydim[0] {
		panic("Unhot: invalid array shape")
	}
	return args(C.UNHOT, xdim[1], xdim[0], x.Data(), y.Data())
}

// Scale array elementwise
func Scale(alpha float32, x Array) Function {
	if x.Dtype() != Float32 {
		panic("Axpy: dtype must by Float32")
	}
	n := Prod(x.Dims())
	return args(C.SCALE, n, alpha, x.Data())
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

// Calculate the scalar sum of the values in the array.
func Sum(a, total Array) Function {
	if len(total.Dims()) != 0 || total.Dtype() != Float32 {
		panic("Sum: result type should be float32 scalar")
	}
	return args(C.SUM, int(a.Dtype()), Prod(a.Dims()), a.Data(), total.Data())
}

// Element wise array multiplication: c = a*b
func Mul(a, b, c Array) Function {
	asize, bsize, csize := Prod(a.Dims()), Prod(b.Dims()), Prod(c.Dims())
	if asize != csize || bsize != csize {
		panic("Mul: arrays must be same size")
	}
	return args(C.MUL_ELEM, asize, a.Data(), b.Data(), c.Data())
}

// Matrix vector multiplication: y <- dot(mA,x)
func Gemv(mA, x, y Array, aTrans TransType) Function {
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
	return args(C.GEMV, int(aTrans), m, n, mA.Data(), x.Data(), y.Data())
}

// Matrix matrix multiplication: mC <- dot(mA, mB) or mC <- dot(mA, mB) + mC if incr = true
func Gemm(mA, mB, mC Array, aTrans, bTrans TransType, incr bool) Function {
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
	var beta float32
	if incr {
		beta = 1
	}
	return args(C.GEMM, int(aTrans), int(bTrans), m, n, k, adim[0], bdim[0], cdim[0],
		beta, mA.Data(), mB.Data(), mC.Data())
}

// Quadratic loss function: (x-y)**2
func QuadraticLoss(x, y, res Array) Function {
	if x.Dtype() != Float32 || y.Dtype() != Float32 || res.Dtype() != Float32 {
		panic("QuadraticLoss: dtype must by Float32")
	}
	if !SameShape(x.Dims(), res.Dims()) || !SameShape(y.Dims(), res.Dims()) {
		panic("QuadraticLoss: arrays must be same shape")
	}
	return args(C.QUAD_LOSS, Prod(x.Dims()), x.Data(), y.Data(), res.Data())
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

// Function which may be called via the queue
type Function struct {
	args *C.struct_args
}

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
		case string:
			a.desc = unsafe.Pointer(&v)
		default:
			panic(fmt.Sprintf("invalid arg type: %T", val))
		}
	}
	return Function{args: a}
}

func (f Function) String() string {
	s := getOpName(int(f.args.op))
	for i, ival := range f.args.i {
		s += fmt.Sprintf(" i%d=%d", i, ival)
	}
	for i, fval := range f.args.f {
		s += fmt.Sprintf(" f%d=%g", i, fval)
	}
	for i, pval := range f.args.p {
		s += fmt.Sprintf(" p%d=%x", i, pval)
	}
	return s
}

func (f Function) setData(arr ...Array) Function {
	for i, a := range arr {
		f.args.p[i] = a.Data()
	}
	return f
}

func opDesc(f Function) string {
	if f.args.desc != nil {
		return *((*string)(f.args.desc))
	}
	return getOpName(int(f.args.op))
}

func ptr(ival interface{}) unsafe.Pointer {
	v := reflect.ValueOf(ival)
	return unsafe.Pointer(v.Pointer())
}
