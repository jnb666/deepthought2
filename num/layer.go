package num

/*
#include "num.h"
*/
import "C"

import (
	"fmt"
	"github.com/jnb666/deepthought2/num/cuda"
	"github.com/jnb666/deepthought2/num/mkl"
	"math/rand"
	"unsafe"
)

type LayerOpts int

const (
	FpropOnly      LayerOpts = 0
	BpropData      LayerOpts = 1
	BpropWeights   LayerOpts = 2
	MomentumUpdate LayerOpts = 4
)

func (l LayerOpts) String() string {
	s := "Fprop"
	for i, name := range []string{"BpropData", "BpropWeights", "Momentum"} {
		if l&(1<<uint(i)) != 0 {
			s += "|" + name
		}
	}
	return s
}

// Layer interface type represents an Activation or MaxPool layer
type Layer interface {
	InShape() []int
	OutShape() []int
	Fprop(q Queue, in *Array, work *Pool, trainMode bool) *Array
	Bprop(q Queue, grad, dsrc *Array, work *Pool) *Array
	Output() *Array
	Memory() (weights, outputs, temp int)
	Release()
}

// Param layer also has weights and biases
type ParamLayer interface {
	Layer
	BiasShape() []int
	FilterShape() []int
	SetParamData(W, B, dW, dB *Array)
}

// BatchNorm layer has extra parameters
type BatchNormLayer interface {
	ParamLayer
	InitParams(q Queue)
	Stats() (runMean, runVar *Array)
}

// Create new convolution layer, input shape is nBatch x depth x h x w, returns workspace needed in 32 bit words
func NewConvLayer(q Queue, opts LayerOpts, inShape []int, nFeats, size, stride int, pad, noBias bool) (ParamLayer, int) {
	if len(inShape) != 4 {
		panic("ConvLayer: expect 4 dimensional input")
	}
	n, c, h, w := inShape[3], inShape[2], inShape[1], inShape[0]
	switch d := q.Dev().(type) {
	case cpuDevice:
		layer := mkl.Convolution(d.attr, n, c, h, w, nFeats, size, stride, pad, noBias)
		return newLayerMKL(layer), 0
	case gpuDevice:
		layer := cuda.Convolution(n, c, h, w, nFeats, size, stride, pad, noBias)
		workSize := layer.Init(q.(*gpuQueue).stream, opts&BpropWeights != 0, opts&BpropData != 0)
		l := &convCuda{
			ConvLayer: layer,
			layerBase: newLayerBase(d, layer.InShape(), layer.OutShape()),
			opts:      opts,
		}
		return l, workSize
	default:
		panic("device type not supported")
	}
}

type convCuda struct {
	*cuda.ConvLayer
	*layerBase
	w, b   unsafe.Pointer
	dw, db unsafe.Pointer
	opts   LayerOpts
}

func (l *convCuda) Release() {
	l.ConvLayer.Release()
	l.layerBase.Release()
}

func (l *convCuda) SetParamData(W, B, dW, dB *Array) {
	l.w, l.dw = W.Data(), dW.Data()
	if l.BiasShape() != nil {
		l.b, l.db = B.Data(), dB.Data()
	}
}

func (l *convCuda) Fprop(que Queue, in *Array, work *Pool, trainMode bool) *Array {
	if !SameShape(in.Dims, l.InShape()) {
		panic(fmt.Errorf("fprop conv: invalid input shape: have %v, expect %v", in.Dims, l.InShape()))
	}
	l.src = in
	q := que.(*gpuQueue)
	q.Call(
		args(C.CUDNN_EXECUTE+cuda.ConvFprop, l.Algo[cuda.FwdAlgo], work.Size()*4, l.Ptr(), work.Data(),
			l.Filter.Ptr(), l.Src.Ptr(), l.Dst.Ptr(), l.w, in.Data(), l.dst.Data()),
	)
	if l.Bias != nil {
		q.Call(
			args(C.CUDNN_EXECUTE+cuda.ConvFpropBias, l.Bias.Ptr(), l.Dst.Ptr(), l.b, l.dst.Data()),
		)
	}
	return l.dst
}

func (l *convCuda) Bprop(que Queue, grad, dsrc *Array, work *Pool) *Array {
	if !SameShape(grad.Dims, l.OutShape()) {
		panic(fmt.Errorf("bprop conv: invalid input shape: have %v, expect %v", grad.Dims, l.OutShape()))
	}
	q := que.(*gpuQueue)
	if l.Bias != nil {
		q.Call(
			args(C.CUDNN_EXECUTE+cuda.ConvBpropBias, l.Dst.Ptr(), l.Bias.Ptr(), grad.Data(), l.db),
		)
	}
	q.Call(
		args(C.CUDNN_EXECUTE+cuda.ConvBpropFilter, l.Algo[cuda.BwdFilterAlgo], work.Size()*4, l.Ptr(), work.Data(),
			l.Src.Ptr(), l.Dst.Ptr(), l.Filter.Ptr(), l.src.Data(), grad.Data(), l.dw),
	)
	if dsrc != nil {
		q.Call(
			args(C.CUDNN_EXECUTE+cuda.ConvBpropData, l.Algo[cuda.BwdDataAlgo], work.Size()*4, l.Ptr(), work.Data(),
				l.Filter.Ptr(), l.Dst.Ptr(), l.Src.Ptr(), l.w, grad.Data(), dsrc.Data()),
		)
	}
	return dsrc
}

// Create new max pooling layer, prev layer should be a ConvLayer
func NewPoolLayer(q Queue, opts LayerOpts, inShape []int, size, stride int, pad, average bool) Layer {
	if len(inShape) != 4 {
		panic("PoolLayer: expect 4 dimensional input")
	}
	n, c, h, w := inShape[3], inShape[2], inShape[1], inShape[0]
	switch d := q.Dev().(type) {
	case cpuDevice:
		layer := mkl.Pooling(d.attr, n, c, h, w, size, stride, pad, average)
		return newLayerMKL(layer)
	case gpuDevice:
		layer := cuda.Pooling(n, c, h, w, size, stride, pad, average)
		return &poolCuda{
			PoolLayer: layer,
			layerBase: newLayerBase(d, layer.InShape(), layer.OutShape()),
		}
	default:
		panic("device type not supported")
	}
}

type poolCuda struct {
	*cuda.PoolLayer
	*layerBase
}

func (l *poolCuda) Release() {
	l.PoolLayer.Release()
	l.layerBase.Release()
}

func (l *poolCuda) Fprop(q Queue, in *Array, work *Pool, trainMode bool) *Array {
	if !SameShape(in.Dims, l.InShape()) {
		panic(fmt.Errorf("fprop pool: invalid input shape: have %v, expect %v", in.Dims, l.InShape()))
	}
	l.src = in
	q.Call(
		args(C.CUDNN_EXECUTE+cuda.PoolFprop, l.Ptr(), l.Src.Ptr(), l.Dst.Ptr(), in.Data(), l.dst.Data()),
	)
	return l.dst
}

func (l *poolCuda) Bprop(q Queue, grad, dsrc *Array, work *Pool) *Array {
	if !SameShape(grad.Dims, l.OutShape()) {
		panic(fmt.Errorf("bprop pool: invalid input shape: have %v, expect %v", grad.Dims, l.OutShape()))
	}
	q.Call(
		args(C.CUDNN_EXECUTE+cuda.PoolBprop, l.Ptr(), l.Src.Ptr(), l.Dst.Ptr(), l.dst.Data(), grad.Data(), l.src.Data(), dsrc.Data()),
	)
	return dsrc
}

// Create new activation layer, typ may be sigmoid, tanh or relu
func NewActivationLayer(q Queue, typ string, shape []int) Layer {
	if typ == "softmax" {
		return newActivation(q.Dev(), C.SOFTMAX, -1, shape)
	}
	switch d := q.Dev().(type) {
	case cpuDevice:
		switch typ {
		case "sigmoid":
			return newActivation(d, C.SIGMOID, C.SIGMOID_D, shape)
		case "tanh":
			return newActivation(d, C.TANH, C.TANH_D, shape)
		case "relu":
			return newActivation(d, C.RELU, C.RELU_D, shape)
		default:
			panic("ActivationLayer: type " + typ + " not supported")
		}
	case gpuDevice:
		return &activationCuda{
			ActivLayer: cuda.Activation(typ, shape),
			layerBase:  newLayerBase(d, shape, shape),
		}
	default:
		panic("device type not supported")
	}
}

type activationCuda struct {
	*cuda.ActivLayer
	*layerBase
}

func (l *activationCuda) Release() {
	l.ActivLayer.Release()
	l.layerBase.Release()
}

func (l *activationCuda) Fprop(q Queue, in *Array, work *Pool, trainMode bool) *Array {
	l.src = in
	q.Call(
		args(C.CUDNN_EXECUTE+cuda.ActivFprop, l.Ptr(), l.Src.Ptr(), in.Data(), l.dst.Data()),
	)
	return l.dst
}

func (l *activationCuda) Bprop(q Queue, grad, dsrc *Array, work *Pool) *Array {
	q.Call(
		args(C.CUDNN_EXECUTE+cuda.ActivBprop, l.Ptr(), l.Src.Ptr(), l.dst.Data(), grad.Data(), l.src.Data(), dsrc.Data()),
	)
	return dsrc
}

type activation struct {
	*layerBase
	fwd, bwd Function
	softmax  bool
}

func newActivation(dev Device, fwd, bwd int, shape []int) *activation {
	size := Prod(shape)
	a := &activation{layerBase: newLayerBase(dev, shape, shape)}
	if fwd == C.SOFTMAX {
		a.softmax = true
	} else {
		a.fwd = args(fwd, size)
		a.bwd = args(bwd, size)
	}
	return a
}

func (a *activation) InShape() []int { return a.dst.Dims }

func (a *activation) OutShape() []int { return a.dst.Dims }

func (a *activation) Fprop(q Queue, in *Array, work *Pool, trainMode bool) *Array {
	a.src = in
	if a.softmax {
		q.Call(Softmax(a.src, a.dst))
	} else {
		q.Call(a.fwd.setData(a.src, a.dst))
	}
	return a.dst
}

func (a *activation) Bprop(q Queue, grad, dsrc *Array, work *Pool) *Array {
	if a.softmax {
		q.Call(Copy(grad, dsrc))
	} else {
		q.Call(a.bwd.setData(a.src, grad, dsrc))
	}
	return dsrc
}

// Create new dropout layer.
func NewDropoutLayer(q Queue, ratio float64, shape []int, seed int64) Layer {
	switch q := q.(type) {
	case *cpuQueue:
		return &dropout{
			ratio:     ratio,
			layerBase: newLayerBase(q.Dev(), shape, shape),
			filter:    q.NewArray(Float32, shape...),
			mask:      make([]float32, Prod(shape)),
			rng:       rand.New(rand.NewSource(seed)),
		}
	case *gpuQueue:
		return &dropoutCuda{
			DropoutLayer: cuda.Dropout(q.stream, ratio, shape, seed),
			layerBase:    newLayerBase(q.Dev(), shape, shape),
		}
	default:
		panic("device type not supported")
	}
}

type dropoutCuda struct {
	*cuda.DropoutLayer
	*layerBase
}

func (l *dropoutCuda) Memory() (weights, output, temp int) {
	return 0, Bytes(l.dst), l.Reserve.Size()*4 + l.States.Size()*4
}

func (l *dropoutCuda) Release() {
	l.layerBase.Release()
	l.DropoutLayer.Release()
}

func (l *dropoutCuda) Fprop(q Queue, in *Array, work *Pool, trainMode bool) *Array {
	if !trainMode {
		return in
	}
	l.src = in
	q.Call(
		args(C.CUDNN_EXECUTE+cuda.DropoutFprop, l.Ptr(), l.Src.Ptr(), in.Data(), l.dst.Data(), l.Reserve.Data(), l.Reserve.Size()*4),
	)
	return l.dst
}

func (l *dropoutCuda) Bprop(q Queue, grad, dsrc *Array, work *Pool) *Array {
	q.Call(
		args(C.CUDNN_EXECUTE+cuda.DropoutBprop, l.Ptr(), l.Src.Ptr(), grad.Data(), dsrc.Data(), l.Reserve.Data(), l.Reserve.Size()*4),
	)
	return dsrc
}

type dropout struct {
	*layerBase
	ratio  float64
	filter *Array
	mask   []float32
	rng    *rand.Rand
}

func (l *dropout) Memory() (weights, output, temp int) {
	return 0, Bytes(l.dst), Bytes(l.filter) + 4*len(l.mask)
}

func (l *dropout) InShape() []int { return l.dst.Dims }

func (l *dropout) OutShape() []int { return l.dst.Dims }

func (l *dropout) Fprop(q Queue, in *Array, work *Pool, trainMode bool) *Array {
	if !trainMode {
		return in
	}
	for i := range l.mask {
		if l.rng.Float64() < l.ratio {
			l.mask[i] = 0
		} else {
			l.mask[i] = 1
		}
	}
	q.Call(
		Write(l.filter, l.mask),
		Mul(l.filter, in, l.dst),
	)
	return l.dst
}

func (l *dropout) Bprop(q Queue, grad, dsrc *Array, work *Pool) *Array {
	q.Call(Mul(l.filter, grad, dsrc))
	return dsrc
}

// Create new batch normalisation layer
func NewBatchNormLayer(q Queue, opts LayerOpts, avgFactor, epsilon float64, shape []int) BatchNormLayer {
	if len(shape) != 4 {
		panic("BatchNormLayer: expect 4 dimensional input")
	}
	n, c, h, w := shape[3], shape[2], shape[1], shape[0]
	p := batchNorm{
		avgFactor: float32(avgFactor),
		epsilon:   float32(epsilon),
		mean:      q.NewArray(Float32, c),
		variance:  q.NewArray(Float32, c),
		runMean:   q.NewArray(Float32, c),
		runVar:    q.NewArray(Float32, c),
	}
	switch d := q.Dev().(type) {
	case cpuDevice:
		return &batchNormMkl{
			batchNorm: p,
			layerMKL:  newLayerMKL(mkl.BatchNorm(d.attr, n, c, h, w, epsilon)),
		}
	case gpuDevice:
		return &batchNormCuda{
			batchNorm:      p,
			BatchNormLayer: cuda.BatchNorm(n, c, h, w),
			layerBase:      newLayerBase(d, shape, shape),
		}
	default:
		panic("device type not supported")
	}
}

type batchNorm struct {
	avgFactor float32
	epsilon   float32
	w, b      *Array
	dw, db    *Array
	mean      *Array
	variance  *Array
	runMean   *Array
	runVar    *Array
}

func (l batchNorm) Stats() (runMean, runVar *Array) {
	return l.runMean, l.runVar
}

func (l batchNorm) memory() int {
	return Bytes(l.mean, l.variance, l.runMean, l.runVar)
}

func (l batchNorm) release() {
	Release(l.mean, l.variance, l.runMean, l.runVar)
}

type batchNormCuda struct {
	batchNorm
	*cuda.BatchNormLayer
	*layerBase
}

func (l *batchNormCuda) SetParamData(W, B, dW, dB *Array) {
	l.w, l.b, l.dw, l.db = W, B, dW, dB
}

func (l *batchNormCuda) InitParams(q Queue) {
	q.Call(
		Fill(l.w, 1),
		Fill(l.b, 0),
		Fill(l.runMean, 0),
		Fill(l.runVar, 1),
	)
}

func (l *batchNormCuda) Release() {
	l.batchNorm.release()
	l.BatchNormLayer.Release()
	l.layerBase.Release()
}

func (l *batchNormCuda) Memory() (weights, output, temp int) {
	return 0, Bytes(l.dst), l.memory()
}

func (l *batchNormCuda) Fprop(q Queue, in *Array, work *Pool, trainMode bool) *Array {
	f := C.CUDNN_EXECUTE + cuda.BnormFpropInfer
	if trainMode {
		f = C.CUDNN_EXECUTE + cuda.BnormFpropTrain
		l.src = in
	}
	q.Call(
		args(f, l.Src.Ptr(), in.Data(), l.dst.Data(), l.Shape.Ptr(), l.w.Data(), l.b.Data(),
			l.runMean.Data(), l.runVar.Data(), l.mean.Data(), l.variance.Data(),
			l.epsilon, l.avgFactor),
	)
	return l.dst
}

func (l *batchNormCuda) Bprop(q Queue, grad, dsrc *Array, work *Pool) *Array {
	q.Call(
		args(C.CUDNN_EXECUTE+cuda.BnormBprop, l.Src.Ptr(), l.src.Data(), grad.Data(), dsrc.Data(),
			l.Shape.Ptr(), l.w.Data(), l.dw.Data(), l.db.Data(), l.mean.Data(), l.variance.Data(),
			l.epsilon),
	)
	return dsrc
}

type batchNormMkl struct {
	batchNorm
	layerMKL
}

func (l *batchNormMkl) SetParamData(W, B, dW, dB *Array) {
	l.w, l.dw = W, dW
	l.SetStatsData(l.w.Data(), l.dw.Data(), l.mean.Data(), l.variance.Data())
}

func (l *batchNormMkl) InitParams(q Queue) {
	q.Call(
		Fill(l.w, 0),
		WriteCol(l.w, 0, ones(l.w.Dims[0])),
		Fill(l.runMean, 0),
		Fill(l.runVar, 1),
	)
}

func (l *batchNormMkl) Release() {
	l.batchNorm.release()
	l.layerMKL.Release()
}

func (l *batchNormMkl) Memory() (weights, output, temp int) {
	return 0, Bytes(l.dst), l.memory()
}

func (l *batchNormMkl) Bprop(q Queue, grad, dsrc *Array, work *Pool) *Array {
	l.layerMKL.Bprop(q, grad, dsrc, work)
	q.Call(
		Axpy(1-l.avgFactor, l.runMean, l.runMean),
		Axpy(l.avgFactor, l.mean, l.runMean),
		Axpy(1-l.avgFactor, l.runVar, l.runVar),
		Axpy(l.avgFactor, l.variance, l.runVar),
	)
	return dsrc
}

// layer which wraps Intel MKL DNN layer
type layerMKL struct {
	*mkl.Layer
	dst *Array
}

func newLayerMKL(layer *mkl.Layer) layerMKL {
	l := layerMKL{
		Layer: layer,
		dst:   &Array{Buffer: layer.Dst(), Dtype: Float32, Dims: layer.OutShape()},
	}
	return l
}

func (l layerMKL) Memory() (weights, output, temp int) {
	return 0, Bytes(l.dst), l.Worksize() * 4
}

func (l layerMKL) SetParamData(W, B, dW, dB *Array) {
	if l.BiasShape() != nil {
		l.Layer.SetParams(W.Data(), B.Data(), dW.Data(), dB.Data())
	} else {
		l.Layer.SetParams(W.Data(), nil, dW.Data(), nil)
	}
}

func (l layerMKL) Output() *Array { return l.dst }

func (l layerMKL) Fprop(q Queue, in *Array, work *Pool, trainMode bool) *Array {
	if !SameShape(in.Dims, l.InShape()) {
		panic(fmt.Errorf("fprop: invalid input shape: have %v, expect %v", in.Dims, l.InShape()))
	}
	l.SetSrc(in.Data())
	q.Call(dnnExecute(l.Fwd, l.ResPtr(), l.Type()+"_fprop"))
	return l.dst
}

func (l layerMKL) Bprop(q Queue, grad, dsrc *Array, work *Pool) *Array {
	if !SameShape(grad.Dims, l.OutShape()) {
		panic(fmt.Errorf("bprop: invalid input shape: have %v, expect %v", grad.Dims, l.OutShape()))
	}
	l.SetDiffDst(grad.Data())
	if l.BBias != nil {
		q.Call(dnnExecute(l.BBias, l.ResPtr(), l.Type()+"_bprop_bias"))
	}
	if l.BFilter != nil {
		q.Call(dnnExecute(l.BFilter, l.ResPtr(), l.Type()+"_bprop_filter"))
	}
	if dsrc != nil {
		l.SetDiffSrc(dsrc.Data())
		q.Call(dnnExecute(l.BData, l.ResPtr(), l.Type()+"_bprop_data"))
	}
	return dsrc
}

func dnnExecute(p *mkl.Primitive, res unsafe.Pointer, desc string) Function {
	if p == nil || p.Ptr() == nil {
		panic("dnnExecute: primitive is nil")
	}
	if res == nil {
		panic("dnnExecute: resource pointer is nil")
	}
	return args(C.MKL_DNN_EXECUTE, p.Ptr(), res, desc)
}

// base layer type
type layerBase struct {
	inShape  []int
	src, dst *Array
}

func newLayerBase(d Device, inShape, outShape []int) *layerBase {
	return &layerBase{
		inShape: inShape,
		dst:     d.NewArray(Float32, outShape...),
	}
}

func (l *layerBase) Output() *Array { return l.dst }

func (l *layerBase) Memory() (weights, output, temp int) {
	return 0, Bytes(l.dst), 0
}

func (l *layerBase) Release() {
	Release(l.dst)
}

func ones(n int) []float32 {
	arr := make([]float32, n)
	for i := range arr {
		arr[i] = 1
	}
	return arr
}
