package cuda

/*
#include <cudnn.h>
*/
import "C"

import (
	"fmt"
	"runtime"
	"unsafe"
)

const (
	ActivFprop = iota
	ActivBprop
	DropoutFprop
	DropoutBprop
	ConvFprop
	ConvFpropBias
	ConvBpropData
	ConvBpropFilter
	ConvBpropBias
	PoolFprop
	PoolBprop
)

const (
	FwdAlgo = iota
	BwdFilterAlgo
	BwdDataAlgo
)

var opName = map[int]string{
	ActivFprop:      "activ_fprop",
	ActivBprop:      "activ_bprop",
	DropoutFprop:    "dropout_fprop",
	DropoutBprop:    "dropout_bprop",
	ConvFprop:       "conv_fprop",
	ConvFpropBias:   "conv_fprop_bias",
	ConvBpropData:   "conv_bprop_data",
	ConvBpropFilter: "conv_bprop_filter",
	ConvBpropBias:   "conv_bprop_bias",
	PoolFprop:       "pool_fprop",
	PoolBprop:       "pool_bprop",
}

func OpName(op int) string {
	return opName[op]
}

var activationTypes = map[string]C.cudnnActivationMode_t{
	"sigmoid": C.CUDNN_ACTIVATION_SIGMOID,
	"tanh":    C.CUDNN_ACTIVATION_TANH,
	"relu":    C.CUDNN_ACTIVATION_RELU,
}

// Convolution layer descriptor
type ConvLayer struct {
	Src    *Layout
	Dst    *Layout
	Bias   *Layout
	Filter *FilterLayout
	Algo   [3]int
	desc   C.cudnnConvolutionDescriptor_t
	freed  bool
}

// Create new convolution layer
func Convolution(s *Stream, n, c, h, w, nFeats, filtSize, stride, pad int) (*ConvLayer, int) {
	l := &ConvLayer{}
	wOut := outSize(w, filtSize, stride, pad)
	hOut := outSize(h, filtSize, stride, pad)
	l.Src = NewLayout(n, c, h, w)
	l.Dst = NewLayout(n, nFeats, hOut, wOut)
	l.Filter = NewFilterLayout(nFeats, c, filtSize, filtSize)
	l.Bias = NewLayout(1, nFeats, 1, 1)
	chkDnn(C.cudnnCreateConvolutionDescriptor(&l.desc))
	chkDnn(C.cudnnSetConvolution2dDescriptor(l.desc, C.int(pad), C.int(pad), C.int(stride), C.int(stride),
		1, 1, C.CUDNN_CROSS_CORRELATION, C.CUDNN_DATA_FLOAT))
	runtime.SetFinalizer(l, func(obj *ConvLayer) { obj.Release() })
	workSize := l.init(s)
	return l, workSize
}

// Initialise the layer, returns work space size needed
func (l *ConvLayer) init(s *Stream) int {
	var size [3]C.size_t
	var fwdAlgo C.cudnnConvolutionFwdAlgo_t
	chkDnn(C.cudnnGetConvolutionForwardAlgorithm(s.cudnn, l.Src.desc, l.Filter.desc, l.desc, l.Dst.desc,
		C.CUDNN_CONVOLUTION_FWD_PREFER_FASTEST, 0, &fwdAlgo))
	chkDnn(C.cudnnGetConvolutionForwardWorkspaceSize(s.cudnn, l.Src.desc, l.Filter.desc, l.desc, l.Dst.desc, fwdAlgo, &size[0]))
	l.Algo[FwdAlgo] = int(fwdAlgo)
	var bFiltAlgo C.cudnnConvolutionBwdFilterAlgo_t
	chkDnn(C.cudnnGetConvolutionBackwardFilterAlgorithm(s.cudnn, l.Src.desc, l.Dst.desc, l.desc, l.Filter.desc,
		C.CUDNN_CONVOLUTION_BWD_FILTER_PREFER_FASTEST, 0, &bFiltAlgo))
	chkDnn(C.cudnnGetConvolutionBackwardFilterWorkspaceSize(s.cudnn, l.Src.desc, l.Dst.desc, l.desc, l.Filter.desc, bFiltAlgo, &size[1]))
	l.Algo[BwdFilterAlgo] = int(BwdFilterAlgo)
	var bDataAlgo C.cudnnConvolutionBwdDataAlgo_t
	chkDnn(C.cudnnGetConvolutionBackwardDataAlgorithm(s.cudnn, l.Filter.desc, l.Dst.desc, l.desc, l.Src.desc,
		C.CUDNN_CONVOLUTION_BWD_DATA_PREFER_FASTEST, 0, &bDataAlgo))
	chkDnn(C.cudnnGetConvolutionBackwardDataWorkspaceSize(s.cudnn, l.Filter.desc, l.Dst.desc, l.desc, l.Src.desc, bDataAlgo, &size[2]))
	l.Algo[BwdDataAlgo] = int(bDataAlgo)
	return maxSize(size[:])
}

func (l *ConvLayer) InShape() []int { return l.Src.Dims }

func (l *ConvLayer) OutShape() []int { return l.Dst.Dims }

func (l *ConvLayer) FilterShape() []int { return l.Filter.Dims }

func (l *ConvLayer) Ptr() unsafe.Pointer {
	return unsafe.Pointer(l.desc)
}

func (l *ConvLayer) Release() {
	if !l.freed {
		C.cudnnDestroyConvolutionDescriptor(l.desc)
		l.freed = true
	}
}

// Max pool layer description
type PoolLayer struct {
	Src   *Layout
	Dst   *Layout
	desc  C.cudnnPoolingDescriptor_t
	freed bool
}

// Setup new max pooling layer
func MaxPooling(n, c, h, w, size, stride int) *PoolLayer {
	l := &PoolLayer{}
	wOut := outSize(w, size, stride, 0)
	hOut := outSize(h, size, stride, 0)
	l.Src = NewLayout(n, c, h, w)
	l.Dst = NewLayout(n, c, hOut, wOut)

	chkDnn(C.cudnnCreatePoolingDescriptor(&l.desc))
	chkDnn(C.cudnnSetPooling2dDescriptor(l.desc, C.CUDNN_POOLING_MAX, C.CUDNN_PROPAGATE_NAN,
		C.int(size), C.int(size), 0, 0, C.int(stride), C.int(stride)))

	runtime.SetFinalizer(l, func(obj *PoolLayer) { obj.Release() })
	return l
}

func (l *PoolLayer) InShape() []int { return l.Src.Dims }

func (l *PoolLayer) OutShape() []int { return l.Dst.Dims }

func (l *PoolLayer) Ptr() unsafe.Pointer {
	return unsafe.Pointer(l.desc)
}

func (l *PoolLayer) Release() {
	if !l.freed {
		C.cudnnDestroyPoolingDescriptor(l.desc)
		l.freed = true
	}
}

// Activation layer descriptor
type ActivLayer struct {
	Src   *Layout
	dims  []int
	desc  C.cudnnActivationDescriptor_t
	freed bool
}

// Create new activation layer
func Activation(typ string, shape []int) *ActivLayer {
	l := &ActivLayer{dims: shape}
	mode, ok := activationTypes[typ]
	if !ok {
		panic("cuDNN: activation type " + typ + " is not valid")
	}
	if len(shape) == 4 {
		l.Src = NewLayout(shape[3], shape[2], shape[1], shape[0])
	} else if len(shape) == 2 {
		l.Src = NewLayout(shape[1], shape[0], 1, 1)
	} else {
		panic("cuDNN: activation layer must have 2 or 4 dimensions")
	}
	chkDnn(C.cudnnCreateActivationDescriptor(&l.desc))
	chkDnn(C.cudnnSetActivationDescriptor(l.desc, mode, C.CUDNN_PROPAGATE_NAN, 0.0))
	runtime.SetFinalizer(l, func(obj *ActivLayer) { obj.Release() })
	return l
}

func (l *ActivLayer) InShape() []int { return l.dims }

func (l *ActivLayer) OutShape() []int { return l.dims }

func (l *ActivLayer) Ptr() unsafe.Pointer {
	return unsafe.Pointer(l.desc)
}

func (l *ActivLayer) Release() {
	if !l.freed {
		C.cudnnDestroyActivationDescriptor(l.desc)
		l.freed = true
	}
}

// Dropout layer descriptor
type DropoutLayer struct {
	Src     *Layout
	Reserve Buffer
	states  Buffer
	dims    []int
	desc    C.cudnnDropoutDescriptor_t
	freed   bool
}

// Create new dropout layer
func Dropout(s *Stream, ratio float64, shape []int, seed int64) *DropoutLayer {
	l := &DropoutLayer{dims: shape}
	if len(shape) == 4 {
		l.Src = NewLayout(shape[3], shape[2], shape[1], shape[0])
	} else if len(shape) == 2 {
		l.Src = NewLayout(shape[1], shape[0], 1, 1)
	} else {
		panic("cuDNN: dropout layer must have 2 or 4 dimensions")
	}
	var size C.size_t
	chkDnn(C.cudnnDropoutGetStatesSize(s.cudnn, &size))
	l.states = NewBuffer(int(size))
	chkDnn(C.cudnnDropoutGetReserveSpaceSize(l.Src.desc, &size))
	l.Reserve = NewBuffer(int(size))

	chkDnn(C.cudnnCreateDropoutDescriptor(&l.desc))
	chkDnn(C.cudnnSetDropoutDescriptor(l.desc, s.cudnn, C.float(ratio),
		l.states.Ptr, C.size_t(l.states.Size), C.ulonglong(seed)))

	runtime.SetFinalizer(l, func(obj *DropoutLayer) { obj.Release() })
	return l
}

func (l *DropoutLayer) InShape() []int { return l.dims }

func (l *DropoutLayer) OutShape() []int { return l.dims }

func (l *DropoutLayer) Ptr() unsafe.Pointer {
	return unsafe.Pointer(l.desc)
}

func (l *DropoutLayer) Release() {
	if !l.freed {
		l.states.Free()
		l.Reserve.Free()
		C.cudnnDestroyDropoutDescriptor(l.desc)
		l.freed = true
	}
}

// Layout type represents a cuDNN tensor descriptor
type Layout struct {
	Dims  []int
	desc  C.cudnnTensorDescriptor_t
	freed bool
}

func NewLayout(n, c, h, w int) *Layout {
	l := &Layout{Dims: []int{w, h, c, n}}
	chkDnn(C.cudnnCreateTensorDescriptor(&l.desc))
	chkDnn(C.cudnnSetTensor4dDescriptor(l.desc, C.CUDNN_TENSOR_NCHW, C.CUDNN_DATA_FLOAT, C.int(n), C.int(c), C.int(h), C.int(w)))
	runtime.SetFinalizer(l, func(obj *Layout) { obj.Release() })
	return l
}

func (l *Layout) Ptr() unsafe.Pointer {
	return unsafe.Pointer(l.desc)
}

func (l *Layout) Release() {
	if !l.freed {
		C.cudnnDestroyTensorDescriptor(l.desc)
		l.freed = true
	}
}

// Filter layout type
type FilterLayout struct {
	Dims  []int
	desc  C.cudnnFilterDescriptor_t
	freed bool
}

func NewFilterLayout(nout, nin, h, w int) *FilterLayout {
	l := &FilterLayout{Dims: []int{w, h, nin, nout}}
	chkDnn(C.cudnnCreateFilterDescriptor(&l.desc))
	chkDnn(C.cudnnSetFilter4dDescriptor(l.desc, C.CUDNN_DATA_FLOAT, C.CUDNN_TENSOR_NCHW, C.int(nout), C.int(nin), C.int(h), C.int(w)))
	runtime.SetFinalizer(l, func(obj *FilterLayout) { obj.Release() })
	return l
}

func (l *FilterLayout) Ptr() unsafe.Pointer {
	return unsafe.Pointer(l.desc)
}

func (l *FilterLayout) Release() {
	if !l.freed {
		C.cudnnDestroyFilterDescriptor(l.desc)
		l.freed = true
	}
}

func outSize(x, size, stride, pad int) int {
	ns := x - size + 2*pad
	if ns%stride != 0 {
		panic("output size invalid, must be even no. of strides")
	}
	return ns/stride + 1
}

func maxSize(size []C.size_t) (max int) {
	for _, s := range size {
		if int(s) > max {
			max = int(s)
		}
	}
	return max
}

type DnnStatus C.cudnnStatus_t

func chkDnn(err C.cudnnStatus_t) {
	if e := GetDnnError(DnnStatus(err)); e != nil {
		panic(e)
	}
}

// Convert cuDNN status code to go error
func GetDnnError(err DnnStatus) error {
	if err == C.CUDNN_STATUS_SUCCESS {
		return nil
	}
	cstr := C.cudnnGetErrorString(C.cudnnStatus_t(err))
	return fmt.Errorf("Cuda error: %s", C.GoString(cstr))
}
