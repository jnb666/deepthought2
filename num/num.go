// Package num contains numeric Array processing routines such as optimised matix multiplication.
package num

/*
#cgo CFLAGS: -g -O2 -std=c99 -I/opt/OpenBLAS/include
#cgo LDFLAGS: -L/opt/OpenBLAS/lib -lopenblas -lm -lpthread
#include "num.h"
*/

/*
#cgo CFLAGS: -g -O2 -std=c99 -I/opt/intel/mkl/include -DUSE_MKL
#cgo LDFLAGS: -L/opt/intel/mkl/lib/intel64 -L/opt/intel/tbb/lib/intel64/gcc4.7 -Wl,--no-as-needed -lmkl_intel_lp64 -lmkl_tbb_thread -lmkl_core -ltbb -lstdc++ -lpthread -lm -ldl
#include "num.h"
*/
import "C"

import (
	"fmt"
	"github.com/jnb666/deepthought2/num/mkl"
	"reflect"
	"unsafe"
)

var opName = map[C.int]string{
	C.COPY:         "copy",
	C.COPY_ROW:     "copy_row",
	C.COPY_COL:     "copy_col",
	C.TILE0:        "tile0",
	C.TILE1:        "tile1",
	C.FILL:         "fill",
	C.NEQ:          "neq",
	C.ONEHOT:       "onehot",
	C.UNHOT:        "unhot",
	C.SCALE:        "scale",
	C.AXPY:         "axpy",
	C.TRANS:        "trams",
	C.SUM:          "sum",
	C.GEMV:         "gemv",
	C.GEMM:         "gemm",
	C.SIGMOID:      "sigmoid",
	C.SIGMOID_D:    "sigmoid_d",
	C.TANH:         "tanh",
	C.TANH_D:       "tanh_d",
	C.RELU:         "relu",
	C.RELU_D:       "relu_d",
	C.QUAD_LOSS:    "quad_loss",
	C.SOFTMAX:      "sofmax",
	C.SOFTMAX_LOSS: "softmax_loss",
	C.DNN_CONVERT:  "dnn_convert",
	C.DNN_EXECUTE:  "dnn_execute",
}

func getOpName(op int) string {
	return opName[C.int(op)]
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
	return args(C.COPY, Prod(a.Dims()), ptr(data), a.Data())
}

// Write data from a slice into the given array.
func Write(a Array, data interface{}) Function {
	return args(C.COPY, Prod(a.Dims()), a.Data(), ptr(data))
}

// Write to one row in the array
func WriteRow(a Array, row int, data interface{}) Function {
	dims := a.Dims()
	if len(dims) != 2 {
		panic("WriteRow: must be a matrix")
	}
	if row < 0 || row >= dims[0] {
		panic("WriteRow: row out of range")
	}
	return args(C.COPY_ROW, row, dims[0], dims[1], a.Data(), ptr(data))
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
func Copy(dst, src Array) Function {
	if src.Dtype() != dst.Dtype() {
		panic("Copy: arguments must be same type")
	}
	ddim, sdim := dst.Dims(), src.Dims()
	if SameShape(ddim, sdim) {
		return args(C.COPY, Prod(ddim), dst.Data(), src.Data())
	} else if len(sdim) == 1 && len(ddim) == 2 && sdim[0] == ddim[1] {
		return args(C.TILE1, sdim[0], ddim[0], dst.Data(), src.Data())
	} else if len(sdim) == 2 && sdim[1] == 1 && len(ddim) == 2 && sdim[0] == ddim[0] {
		return args(C.TILE0, sdim[0], ddim[1], dst.Data(), src.Data())
	} else if len(sdim) == 2 && sdim[0] == 1 && len(ddim) == 2 && sdim[1] == ddim[1] {
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

func dnnConvert(p *mkl.Primitive, src, dst unsafe.Pointer) Function {
	if p == nil || p.Ptr() == nil {
		panic("dnnConvert: primitive is nil")
	}
	if src == nil || dst == nil {
		panic("dnnConvert: input argument is nil")
	}
	return args(C.DNN_CONVERT, p.Ptr(), src, dst)
}

func dnnExecute(p *mkl.Primitive, res unsafe.Pointer, desc string) Function {
	if p == nil || p.Ptr() == nil {
		panic("dnnExecute: primitive is nil")
	}
	if res == nil {
		panic("dnnExecute: resource pointer is nil")
	}
	return args(C.DNN_EXECUTE, p.Ptr(), res, desc)
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
		case string:
			a.desc = unsafe.Pointer(&v)
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
	C.set_num_threads(C.int(threads))
}

func execCPU(buffer []Function, profile bool, p map[string]profileRec) {
	bsize := C.int(len(buffer))
	ptr := (**C.struct_args)(unsafe.Pointer(&buffer[0]))
	var err C.dnnError_t
	var ix C.int
	if profile {
		err = C.execCPUProfile(bsize, ptr, &ix)
		for _, arg := range buffer {
			name := opDesc(arg)
			rec := p[name]
			rec.name = name
			rec.calls++
			rec.usec += int64(arg.usec)
			p[name] = rec
		}
	} else {
		err = C.execCPU(bsize, ptr, &ix)
	}
	if e := mkl.GetError(mkl.Error(err)); e != nil {
		panic(fmt.Sprintf("%s calling %s", e, opDesc(buffer[ix])))
	}
}

func opDesc(arg Function) string {
	if arg.desc != nil {
		return *((*string)(arg.desc))
	}
	return opName[arg.op]
}

func ptr(ival interface{}) unsafe.Pointer {
	v := reflect.ValueOf(ival)
	return unsafe.Pointer(v.Pointer())
}
