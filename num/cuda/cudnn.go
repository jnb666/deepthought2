package cuda

/*
#include <cudnn.h>
*/
import "C"

import (
	"fmt"
	"unsafe"
)

const (
	ActivFprop = iota
	ActivBprop
	DropoutFprop
	DropoutBprop
	BnormFpropInfer
	BnormFpropTrain
	BnormBprop
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
	BnormFpropInfer: "batchnorm_fprop_infer",
	BnormFpropTrain: "batchnorm_fprop_train",
	BnormBprop:      "batchnorm_bprop",
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
func Convolution(n, c, h, w, nFeats, filtSize, stride int, padding, noBias bool) *ConvLayer {
	l := &ConvLayer{}
	wOut, wPad := getOutSize(w, filtSize, stride, padding)
	hOut, hPad := getOutSize(h, filtSize, stride, padding)
	l.Src = NewLayout(n, c, h, w)
	l.Dst = NewLayout(n, nFeats, hOut, wOut)
	l.Filter = NewFilterLayout(nFeats, c, filtSize, filtSize)
	if !noBias {
		l.Bias = NewLayout(1, nFeats, 1, 1)
	}
	chkDnn(C.cudnnCreateConvolutionDescriptor(&l.desc))
	chkDnn(C.cudnnSetConvolution2dDescriptor(l.desc, C.int(hPad), C.int(wPad), C.int(stride), C.int(stride),
		1, 1, C.CUDNN_CROSS_CORRELATION, C.CUDNN_DATA_FLOAT))
	return l
}

// Initialise the layer, returns work space size needed.
func (l *ConvLayer) Init(s *Stream, bpropWeights, bpropData bool) int {
	size := [3]C.size_t{}
	fwdAlgo := l.getFwdAlgo(s)
	chkDnn(C.cudnnGetConvolutionForwardWorkspaceSize(s.cudnn, l.Src.desc, l.Filter.desc, l.desc, l.Dst.desc, fwdAlgo, &size[FwdAlgo]))
	l.Algo[FwdAlgo] = int(fwdAlgo)
	if bpropWeights {
		bFiltAlgo := l.getBwdFilterAlgo(s)
		chkDnn(C.cudnnGetConvolutionBackwardFilterWorkspaceSize(s.cudnn, l.Src.desc, l.Dst.desc, l.desc, l.Filter.desc, bFiltAlgo, &size[BwdFilterAlgo]))
		l.Algo[BwdFilterAlgo] = int(BwdFilterAlgo)
	}
	if bpropData {
		bDataAlgo := l.getBwdDataAlgo(s)
		chkDnn(C.cudnnGetConvolutionBackwardDataWorkspaceSize(s.cudnn, l.Filter.desc, l.Dst.desc, l.desc, l.Src.desc, bDataAlgo, &size[BwdDataAlgo]))
		l.Algo[BwdDataAlgo] = int(bDataAlgo)
	}
	return maxWords(size[:])
}

func (l *ConvLayer) getFwdAlgo(s *Stream) (algo C.cudnnConvolutionFwdAlgo_t) {
	var count C.int
	var perf [C.CUDNN_CONVOLUTION_FWD_ALGO_COUNT]C.cudnnConvolutionFwdAlgoPerf_t
	chkDnn(C.cudnnFindConvolutionForwardAlgorithm(s.cudnn, l.Src.desc, l.Filter.desc, l.desc, l.Dst.desc,
		C.CUDNN_CONVOLUTION_FWD_ALGO_COUNT, &count, &perf[0]))
	chkDnn(perf[0].status)
	return perf[0].algo
}

func (l *ConvLayer) getBwdFilterAlgo(s *Stream) (algo C.cudnnConvolutionBwdFilterAlgo_t) {
	var count C.int
	var perf [C.CUDNN_CONVOLUTION_BWD_FILTER_ALGO_COUNT]C.cudnnConvolutionBwdFilterAlgoPerf_t
	chkDnn(C.cudnnFindConvolutionBackwardFilterAlgorithm(s.cudnn, l.Src.desc, l.Dst.desc, l.desc, l.Filter.desc,
		C.CUDNN_CONVOLUTION_BWD_FILTER_ALGO_COUNT, &count, &perf[0]))
	chkDnn(perf[0].status)
	return perf[0].algo
}

func (l *ConvLayer) getBwdDataAlgo(s *Stream) (algo C.cudnnConvolutionBwdDataAlgo_t) {
	var count C.int
	var perf [C.CUDNN_CONVOLUTION_BWD_DATA_ALGO_COUNT]C.cudnnConvolutionBwdDataAlgoPerf_t
	chkDnn(C.cudnnFindConvolutionBackwardDataAlgorithm(s.cudnn, l.Filter.desc, l.Dst.desc, l.desc, l.Src.desc,
		C.CUDNN_CONVOLUTION_BWD_DATA_ALGO_COUNT, &count, &perf[0]))
	chkDnn(perf[0].status)
	return perf[0].algo
}

func (l *ConvLayer) AlgoName(algo int) string {
	switch algo {
	case FwdAlgo:
		switch l.Algo[algo] {
		case C.CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_GEMM:
			return "FWD_ALGO_IMPLICIT_GEMM"
		case C.CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_PRECOMP_GEMM:
			return "FWD_ALGO_IMPLICIT_PRECOMP_GEMM"
		case C.CUDNN_CONVOLUTION_FWD_ALGO_GEMM:
			return "FWD_ALGO_GEMM"
		case C.CUDNN_CONVOLUTION_FWD_ALGO_DIRECT:
			return "FWD_ALGO_DIRECT"
		case C.CUDNN_CONVOLUTION_FWD_ALGO_FFT:
			return "FWD_ALGO_FFT"
		case C.CUDNN_CONVOLUTION_FWD_ALGO_FFT_TILING:
			return "FWD_ALGO_FFT_TILING"
		case C.CUDNN_CONVOLUTION_FWD_ALGO_WINOGRAD:
			return "FWD_ALGO_WINOGRAD"
		case C.CUDNN_CONVOLUTION_FWD_ALGO_WINOGRAD_NONFUSED:
			return "FWD_ALGO_WINOGRAD_NONFUSED"
		}
	case BwdFilterAlgo:
		switch l.Algo[algo] {
		case C.CUDNN_CONVOLUTION_BWD_FILTER_ALGO_0:
			return "BWD_FILTER_ALGO_0"
		case C.CUDNN_CONVOLUTION_BWD_FILTER_ALGO_1:
			return "BWD_FILTER_ALGO_1"
		case C.CUDNN_CONVOLUTION_BWD_FILTER_ALGO_FFT:
			return "BWD_FILTER_ALGO_FFT"
		case C.CUDNN_CONVOLUTION_BWD_FILTER_ALGO_3:
			return "BWD_FILTER_ALGO_3"
		case C.CUDNN_CONVOLUTION_BWD_FILTER_ALGO_WINOGRAD:
			return "BWD_FILTER_ALGO_WINOGRAD"
		case C.CUDNN_CONVOLUTION_BWD_FILTER_ALGO_WINOGRAD_NONFUSED:
			return "BWD_FILTER_ALGO_WINOGRAD_NONFUSED"
		case C.CUDNN_CONVOLUTION_BWD_FILTER_ALGO_FFT_TILING:
			return "BWD_FILTER_ALGO_FFT_TILING"
		}
	case BwdDataAlgo:
		switch l.Algo[algo] {
		case C.CUDNN_CONVOLUTION_BWD_DATA_ALGO_0:
			return "BWD_DATA_ALGO_0"
		case C.CUDNN_CONVOLUTION_BWD_DATA_ALGO_1:
			return "BWD_DATA_ALGO_1"
		case C.CUDNN_CONVOLUTION_BWD_DATA_ALGO_FFT:
			return "BWD_DATA_ALGO_FFT"
		case C.CUDNN_CONVOLUTION_BWD_DATA_ALGO_FFT_TILING:
			return "BWD_DATA_ALGO_FFT_TILING"
		case C.CUDNN_CONVOLUTION_BWD_DATA_ALGO_WINOGRAD:
			return "BWD_DATA_ALGO_WINOGRAD"
		case C.CUDNN_CONVOLUTION_BWD_DATA_ALGO_WINOGRAD_NONFUSED:
			return "BWD_DATA_ALGO_WINOGRAD_NONFUSED"
		}
	}
	panic("invalid algo type")
}

func (l *ConvLayer) InShape() []int { return l.Src.Dims }

func (l *ConvLayer) OutShape() []int { return l.Dst.Dims }

func (l *ConvLayer) FilterShape() []int { return l.Filter.Dims }

func (l *ConvLayer) BiasShape() []int {
	if l.Bias != nil {
		return l.Bias.Dims
	}
	return nil
}

func (l *ConvLayer) Ptr() unsafe.Pointer {
	return unsafe.Pointer(l.desc)
}

func (l *ConvLayer) Release() {
	if !l.freed {
		l.Src.Release()
		l.Dst.Release()
		l.Filter.Release()
		if l.Bias != nil {
			l.Bias.Release()
		}
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

// Setup new max pooling or average pooling layer
func Pooling(n, c, h, w, size, stride int, padding, average bool) *PoolLayer {
	l := &PoolLayer{}
	wOut, wPad := getOutSize(w, size, stride, padding)
	hOut, hPad := getOutSize(h, size, stride, padding)
	l.Src = NewLayout(n, c, h, w)
	l.Dst = NewLayout(n, c, hOut, wOut)
	var mode C.cudnnPoolingMode_t
	if average {
		mode = C.CUDNN_POOLING_AVERAGE_COUNT_EXCLUDE_PADDING
	} else {
		mode = C.CUDNN_POOLING_MAX
	}
	chkDnn(C.cudnnCreatePoolingDescriptor(&l.desc))
	chkDnn(C.cudnnSetPooling2dDescriptor(l.desc, mode, C.CUDNN_PROPAGATE_NAN,
		C.int(size), C.int(size), C.int(hPad), C.int(wPad), C.int(stride), C.int(stride)))
	return l
}

func (l *PoolLayer) InShape() []int { return l.Src.Dims }

func (l *PoolLayer) OutShape() []int { return l.Dst.Dims }

func (l *PoolLayer) Ptr() unsafe.Pointer {
	return unsafe.Pointer(l.desc)
}

func (l *PoolLayer) Release() {
	if !l.freed {
		l.Src.Release()
		l.Dst.Release()
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
	return l
}

func (l *ActivLayer) InShape() []int { return l.dims }

func (l *ActivLayer) OutShape() []int { return l.dims }

func (l *ActivLayer) Ptr() unsafe.Pointer {
	return unsafe.Pointer(l.desc)
}

func (l *ActivLayer) Release() {
	if !l.freed {
		l.Src.Release()
		C.cudnnDestroyActivationDescriptor(l.desc)
		l.freed = true
	}
}

// Dropout layer descriptor
type DropoutLayer struct {
	Src     *Layout
	Reserve Buffer
	States  Buffer
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
	l.States = NewBuffer(words(size))
	chkDnn(C.cudnnDropoutGetReserveSpaceSize(l.Src.desc, &size))
	l.Reserve = NewBuffer(words(size))

	chkDnn(C.cudnnCreateDropoutDescriptor(&l.desc))
	chkDnn(C.cudnnSetDropoutDescriptor(l.desc, s.cudnn, C.float(ratio),
		l.States.Data(), C.size_t(l.States.Capacity()*4), C.ulonglong(seed)))
	return l
}

func (l *DropoutLayer) InShape() []int { return l.dims }

func (l *DropoutLayer) OutShape() []int { return l.dims }

func (l *DropoutLayer) Ptr() unsafe.Pointer {
	return unsafe.Pointer(l.desc)
}

func (l *DropoutLayer) Release() {
	if !l.freed {
		l.Src.Release()
		l.States.Release()
		l.Reserve.Release()
		C.cudnnDestroyDropoutDescriptor(l.desc)
		l.freed = true
	}
}

// Batch normalisation layer descriptor
type BatchNormLayer struct {
	Src   *Layout
	Shape *Layout
	freed bool
}

// Create new BatchNorm layer
func BatchNorm(n, c, h, w int) *BatchNormLayer {
	l := &BatchNormLayer{}
	l.Src = NewLayout(n, c, h, w)
	l.Shape = NewLayout(1, c, 1, 1)
	return l
}

func (l *BatchNormLayer) InShape() []int { return l.Src.Dims }

func (l *BatchNormLayer) OutShape() []int { return l.Src.Dims }

func (l *BatchNormLayer) FilterShape() []int { return l.Shape.Dims }

func (l *BatchNormLayer) BiasShape() []int { return l.Shape.Dims }

func (l *BatchNormLayer) Release() {
	if !l.freed {
		l.Src.Release()
		l.Shape.Release()
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

// utilities
func getOutSize(in, filter, stride int, padding bool) (out, pad int) {
	if padding {
		out = ciel(in, stride)
		for out > 1+(in-filter+2*pad)/stride {
			pad++
		}
	} else {
		out = 1 + (in-filter)/stride
	}
	return
}

func ciel(a, b int) int {
	if a%b == 0 {
		return a / b
	}
	return 1 + a/b
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
