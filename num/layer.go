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

// Layer interface type represents an Activation or MaxPool layer
type Layer interface {
	InShape() []int
	OutShape() []int
	Fprop(q Queue, in, work Array, trainMode bool) Array
	Bprop(q Queue, grad, work Array) Array
	Output() Array
	Release()
}

// Param layer also has weights and biases
type ParamLayer interface {
	Layer
	FilterShape() []int
	SetParamData(W, B, dW, dB Array)
}

// Create new convolution layer, input shape is nBatch x depth x h x w
func NewConvLayer(q Queue, ix int, inShape []int, nFeats, size, stride, pad int) (ParamLayer, int) {
	if len(inShape) != 4 {
		panic("ConvLayer: expect 4 dimensional input")
	}
	n, c, h, w := inShape[3], inShape[2], inShape[1], inShape[0]
	switch d := q.Dev().(type) {
	case cpuDevice:
		return newLayerMKL(mkl.Convolution(d.attr, n, c, h, w, nFeats, size, stride, pad), ix != 0), 0
	case gpuDevice:
		layer, workSize := cuda.Convolution(q.(*gpuQueue).stream, n, c, h, w, nFeats, size, stride, pad)
		l := &convCuda{
			ConvLayer: layer,
			dst:       d.NewArray(Float32, layer.OutShape()...),
		}
		if ix != 0 {
			l.dSrc = d.NewArray(Float32, layer.InShape()...)
		}
		return l, workSize
	default:
		panic("device type not supported")
	}
}

type convCuda struct {
	*cuda.ConvLayer
	w, b     unsafe.Pointer
	dw, db   unsafe.Pointer
	src, dst Array
	dSrc     Array
}

func (l *convCuda) Release() {
	l.dst.Release()
	if l.dSrc != nil {
		l.dSrc.Release()
	}
	l.ConvLayer.Release()
}

func (l *convCuda) SetParamData(W, B, dW, dB Array) {
	l.w, l.b, l.dw, l.db = W.Data(), B.Data(), dW.Data(), dB.Data()
}

func (l *convCuda) Output() Array { return l.dst }

func (l *convCuda) Fprop(que Queue, in, work Array, trainMode bool) Array {
	if !SameShape(in.Dims(), l.InShape()) {
		panic(fmt.Errorf("fprop conv: invalid input shape: have %v, expect %v", in.Dims(), l.InShape()))
	}
	l.src = in
	q := que.(*gpuQueue)
	q.Call(
		args(C.CUDNN_EXECUTE+cuda.ConvFprop, l.Algo[cuda.FwdAlgo], work.Size(), l.Ptr(), work.Data(),
			l.Filter.Ptr(), l.Src.Ptr(), l.Dst.Ptr(), l.w, in.Data(), l.dst.Data()),
		args(C.CUDNN_EXECUTE+cuda.ConvFpropBias, l.Bias.Ptr(), l.Dst.Ptr(), l.b, l.dst.Data()),
	)
	return l.dst
}

func (l *convCuda) Bprop(que Queue, grad, work Array) Array {
	if !SameShape(grad.Dims(), l.OutShape()) {
		panic(fmt.Errorf("bprop conv: invalid input shape: have %v, expect %v", grad.Dims(), l.OutShape()))
	}
	q := que.(*gpuQueue)
	q.Call(
		args(C.CUDNN_EXECUTE+cuda.ConvBpropBias, l.Dst.Ptr(), l.Bias.Ptr(), grad.Data(), l.db),
		args(C.CUDNN_EXECUTE+cuda.ConvBpropFilter, l.Algo[cuda.BwdFilterAlgo], work.Size(), l.Ptr(), work.Data(),
			l.Src.Ptr(), l.Dst.Ptr(), l.Filter.Ptr(), l.src.Data(), grad.Data(), l.dw),
	)
	if l.dSrc != nil {
		q.Call(
			args(C.CUDNN_EXECUTE+cuda.ConvBpropData, l.Algo[cuda.BwdDataAlgo], work.Size(), l.Ptr(), work.Data(),
				l.Filter.Ptr(), l.Dst.Ptr(), l.Src.Ptr(), l.w, grad.Data(), l.dSrc.Data()),
		)
	}
	return l.dSrc
}

// Create new max pooling layer, prev layer should be a ConvLayer
func NewMaxPoolLayer(q Queue, inShape []int, size, stride int) Layer {
	if len(inShape) != 4 {
		panic("PoolLayer: expect 4 dimensional input")
	}
	n, c, h, w := inShape[3], inShape[2], inShape[1], inShape[0]
	switch d := q.Dev().(type) {
	case cpuDevice:
		return newLayerMKL(mkl.MaxPooling(d.attr, n, c, h, w, size, stride), true)
	case gpuDevice:
		layer := cuda.MaxPooling(n, c, h, w, size, stride)
		return &poolCuda{
			PoolLayer: layer,
			dst:       d.NewArray(Float32, layer.OutShape()...),
			dSrc:      d.NewArray(Float32, layer.InShape()...),
		}
	default:
		panic("device type not supported")
	}
}

type poolCuda struct {
	*cuda.PoolLayer
	src, dst Array
	dSrc     Array
}

func (l *poolCuda) Release() {
	l.dst.Release()
	l.dSrc.Release()
	l.PoolLayer.Release()
}

func (l *poolCuda) Output() Array { return l.dst }

func (l *poolCuda) Fprop(q Queue, in, work Array, trainMode bool) Array {
	if !SameShape(in.Dims(), l.InShape()) {
		panic(fmt.Errorf("fprop pool: invalid input shape: have %v, expect %v", in.Dims(), l.InShape()))
	}
	l.src = in
	q.Call(
		args(C.CUDNN_EXECUTE+cuda.PoolFprop, l.Ptr(), l.Src.Ptr(), l.Dst.Ptr(), in.Data(), l.dst.Data()),
	)
	return l.dst
}

func (l *poolCuda) Bprop(q Queue, grad, work Array) Array {
	if !SameShape(grad.Dims(), l.OutShape()) {
		panic(fmt.Errorf("bprop pool: invalid input shape: have %v, expect %v", grad.Dims(), l.OutShape()))
	}
	q.Call(
		args(C.CUDNN_EXECUTE+cuda.PoolBprop, l.Ptr(), l.Src.Ptr(), l.Dst.Ptr(), l.dst.Data(), grad.Data(), l.src.Data(), l.dSrc.Data()),
	)
	return l.dSrc
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
		layer := cuda.Activation(typ, shape)
		return &activationCuda{
			ActivLayer: layer,
			dst:        d.NewArray(Float32, shape...),
			dSrc:       d.NewArray(Float32, shape...),
		}
	default:
		panic("device type not supported")
	}
}

type activationCuda struct {
	*cuda.ActivLayer
	src, dst Array
	dSrc     Array
}

func (l *activationCuda) Release() {
	l.dst.Release()
	l.dSrc.Release()
	l.ActivLayer.Release()
}

func (l *activationCuda) Output() Array { return l.dst }

func (l *activationCuda) Fprop(q Queue, in, work Array, trainMode bool) Array {
	l.src = in
	q.Call(
		args(C.CUDNN_EXECUTE+cuda.ActivFprop, l.Ptr(), l.Src.Ptr(), in.Data(), l.dst.Data()),
	)
	return l.dst
}

func (l *activationCuda) Bprop(q Queue, grad, work Array) Array {
	q.Call(
		args(C.CUDNN_EXECUTE+cuda.ActivBprop, l.Ptr(), l.Src.Ptr(), l.dst.Data(), grad.Data(), l.src.Data(), l.dSrc.Data()),
	)
	return l.dSrc
}

type activation struct {
	fwd, bwd       Function
	src, dst, dsrc Array
	softmax        bool
}

func newActivation(dev Device, fwd, bwd int, shape []int) *activation {
	size := Prod(shape)
	a := &activation{
		dst:  dev.NewArray(Float32, shape...),
		dsrc: dev.NewArray(Float32, shape...),
	}
	if fwd == C.SOFTMAX {
		a.softmax = true
	} else {
		a.fwd = args(fwd, size)
		a.bwd = args(bwd, size)
	}
	return a
}

func (l *activation) Release() {
	l.dst.Release()
	l.dsrc.Release()
}

func (a *activation) InShape() []int { return a.dst.Dims() }

func (a *activation) OutShape() []int { return a.dst.Dims() }

func (a *activation) Output() Array { return a.dst }

func (a *activation) Fprop(q Queue, in, work Array, trainMode bool) Array {
	a.src = in
	if a.softmax {
		q.Call(Softmax(a.src, a.dst))
	} else {
		q.Call(a.fwd.setData(a.src, a.dst))
	}
	return a.dst
}

func (a *activation) Bprop(q Queue, grad, work Array) Array {
	if a.softmax {
		q.Call(Copy(grad, a.dsrc))
	} else {
		q.Call(a.bwd.setData(a.src, grad, a.dsrc))
	}
	return a.dsrc
}

// Create new dropout layer.
func NewDropoutLayer(q Queue, ratio float64, shape []int, seed int64) Layer {
	switch d := q.Dev().(type) {
	case cpuDevice:
		return &dropout{
			ratio:  ratio,
			dst:    d.NewArray(Float32, shape...),
			dsrc:   d.NewArray(Float32, shape...),
			filter: d.NewArray(Float32, shape...),
			mask:   make([]float32, Prod(shape)),
			rng:    rand.New(rand.NewSource(seed)),
		}
	case gpuDevice:
		layer := cuda.Dropout(q.(*gpuQueue).stream, ratio, shape, seed)
		return &dropoutCuda{
			DropoutLayer: layer,
			dst:          d.NewArray(Float32, shape...),
			dSrc:         d.NewArray(Float32, shape...),
		}
	default:
		panic("device type not supported")
	}
}

type dropoutCuda struct {
	*cuda.DropoutLayer
	dst     Array
	dSrc    Array
	enabled bool
}

func (l *dropoutCuda) Release() {
	l.dst.Release()
	l.dSrc.Release()
	l.DropoutLayer.Release()
}

func (l *dropoutCuda) Output() Array { return l.dst }

func (l *dropoutCuda) Fprop(q Queue, in, work Array, trainMode bool) Array {
	l.enabled = trainMode
	if !l.enabled {
		return in
	}
	q.Call(
		args(C.CUDNN_EXECUTE+cuda.DropoutFprop, l.Ptr(), l.Src.Ptr(), in.Data(), l.dst.Data(), l.Reserve.Ptr, l.Reserve.Size),
	)
	return l.dst
}

func (l *dropoutCuda) Bprop(q Queue, grad, work Array) Array {
	if !l.enabled {
		return grad
	}
	q.Call(
		args(C.CUDNN_EXECUTE+cuda.DropoutBprop, l.Ptr(), l.Src.Ptr(), grad.Data(), l.dSrc.Data(), l.Reserve.Ptr, l.Reserve.Size),
	)
	return l.dSrc
}

type dropout struct {
	ratio     float64
	dst, dsrc Array
	filter    Array
	mask      []float32
	rng       *rand.Rand
}

func (l *dropout) InShape() []int { return l.dst.Dims() }

func (l *dropout) OutShape() []int { return l.dst.Dims() }

func (l *dropout) Output() Array { return l.dst }

func (l *dropout) Release() {
	l.dst.Release()
	l.dsrc.Release()
	l.filter.Release()
}

func (l *dropout) Fprop(q Queue, in, work Array, trainMode bool) Array {
	for i := range l.mask {
		if trainMode && l.rng.Float64() < l.ratio {
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

func (l *dropout) Bprop(q Queue, grad, work Array) Array {
	q.Call(
		Mul(l.filter, grad, l.dsrc),
	)
	return l.dsrc
}

// Create new batch normalisation layer
func NewBatchNormLayer(q Queue, avgFactor, epsilon float64, shape []int, runMean, runVar Array) ParamLayer {
	if len(shape) != 4 {
		panic("BatchNormLayer: expect 4 dimensional input")
	}
	n, c, h, w := shape[3], shape[2], shape[1], shape[0]
	if _, ok := q.Dev().(gpuDevice); !ok {
		panic("BatchNormLayer only implemented for GPU device")
	}
	l := &batchNormCuda{
		BatchNormLayer: cuda.BatchNorm(n, c, h, w),
		avgFactor:      float32(avgFactor),
		epsilon:        float32(epsilon),
		runMean:        runMean.Data(),
		runVar:         runVar.Data(),
		saveMean:       q.NewArray(Float32, c),
		saveStd:        q.NewArray(Float32, c),
		dst:            q.NewArray(Float32, shape...),
		dSrc:           q.NewArray(Float32, shape...),
	}
	return l
}

type batchNormCuda struct {
	*cuda.BatchNormLayer
	avgFactor float32
	epsilon   float32
	w, b      unsafe.Pointer
	dw, db    unsafe.Pointer
	runMean   unsafe.Pointer
	runVar    unsafe.Pointer
	saveMean  Array
	saveStd   Array
	src, dst  Array
	dSrc      Array
}

func (l *batchNormCuda) SetParamData(W, B, dW, dB Array) {
	l.w, l.b, l.dw, l.db = W.Data(), B.Data(), dW.Data(), dB.Data()
}

func (l *batchNormCuda) Release() {
	l.dst.Release()
	l.dSrc.Release()
	l.saveMean.Release()
	l.saveStd.Release()
}

func (l *batchNormCuda) Output() Array { return l.dst }

func (l *batchNormCuda) Fprop(q Queue, in, work Array, trainMode bool) Array {
	f := C.CUDNN_EXECUTE + cuda.BnormFpropInfer
	if trainMode {
		f = C.CUDNN_EXECUTE + cuda.BnormFpropTrain
		l.src = in
	}
	q.Call(
		args(f, l.Src.Ptr(), in.Data(), l.dst.Data(), l.Shape.Ptr(), l.w, l.b,
			l.runMean, l.runVar, l.saveMean.Data(), l.saveStd.Data(),
			l.epsilon, l.avgFactor),
	)
	return l.dst
}

func (l *batchNormCuda) Bprop(q Queue, grad, work Array) Array {
	q.Call(
		args(C.CUDNN_EXECUTE+cuda.BnormBprop, l.Src.Ptr(), l.src.Data(), grad.Data(), l.dSrc.Data(),
			l.Shape.Ptr(), l.w, l.dw, l.db, l.saveMean.Data(), l.saveStd.Data(), l.epsilon),
	)
	return l.dSrc
}

// layer which wraps Intel MKL DNN layer
type layerMKL struct {
	*mkl.Layer
	dst, dSrc Array
}

func newLayerMKL(layer *mkl.Layer, bpropData bool) layerMKL {
	l := layerMKL{
		Layer: layer,
		dst:   newArrayCPU(Float32, layer.OutShape(), layer.Dst()),
	}
	if bpropData {
		l.dSrc = newArrayCPU(Float32, layer.InShape(), layer.DiffSrc())
	}
	return l
}

func (l layerMKL) Release() {}

func (l layerMKL) SetParamData(W, B, dW, dB Array) {
	l.Layer.SetParams(W.Data(), B.Data(), dW.Data(), dB.Data())
}

func (l layerMKL) Output() Array { return l.dst }

func (l layerMKL) Fprop(q Queue, in, work Array, trainMode bool) Array {
	if !SameShape(in.Dims(), l.InShape()) {
		panic(fmt.Errorf("fprop: invalid input shape: have %v, expect %v", in.Dims(), l.InShape()))
	}
	l.SetSrc(in.Data())
	q.Call(
		dnnExecute(l.Fwd, l.ResPtr(), l.Type()+"_fprop"),
	)
	return l.dst
}

func (l layerMKL) Bprop(q Queue, grad, work Array) Array {
	if !SameShape(grad.Dims(), l.OutShape()) {
		panic(fmt.Errorf("bprop: invalid input shape: have %v, expect %v", grad.Dims(), l.OutShape()))
	}
	l.SetDiffDst(grad.Data())
	if l.HasParams() {
		q.Call(
			dnnExecute(l.BBias, l.ResPtr(), l.Type()+"_bprop_bias"),
			dnnExecute(l.BFilter, l.ResPtr(), l.Type()+"_bprop_filter"),
		)
	}
	if l.dSrc != nil {
		q.Call(
			dnnExecute(l.BData, l.ResPtr(), l.Type()+"_bprop_data"),
		)
	}
	return l.dSrc
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
