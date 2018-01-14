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
	Memory() (weights, outputs, temp int)
	Release()
}

// Param layer also has weights and biases
type ParamLayer interface {
	Layer
	FilterShape() []int
	SetParamData(W, B, dW, dB Array)
}

// BatchNorm layer has extra parameters
type BatchNormLayer interface {
	Layer
	Stats() (w, b, runMean, runVar Array)
	UpdateStats(q Queue, learningRate float32)
}

// Create new convolution layer, input shape is nBatch x depth x h x w
func NewConvLayer(q Queue, ix int, inShape []int, nFeats, size, stride int, pad, noBias bool) (ParamLayer, int) {
	if len(inShape) != 4 {
		panic("ConvLayer: expect 4 dimensional input")
	}
	n, c, h, w := inShape[3], inShape[2], inShape[1], inShape[0]
	switch d := q.Dev().(type) {
	case cpuDevice:
		return newLayerMKL(mkl.Convolution(d.attr, n, c, h, w, nFeats, size, stride, pad, noBias), ix != 0), 0
	case gpuDevice:
		layer, workSize := cuda.Convolution(q.(*gpuQueue).stream, n, c, h, w, nFeats, size, stride, pad, noBias)
		l := &convCuda{
			ConvLayer: layer,
			layerBase: newLayerBase(d, layer.InShape(), layer.OutShape()),
			layerId:   ix,
		}
		return l, workSize
	default:
		panic("device type not supported")
	}
}

type convCuda struct {
	*cuda.ConvLayer
	*layerBase
	w, b    unsafe.Pointer
	dw, db  unsafe.Pointer
	layerId int
}

func (l *convCuda) Release() {
	l.ConvLayer.Release()
	l.layerBase.Release()
}

func (l *convCuda) SetParamData(W, B, dW, dB Array) {
	l.w, l.b, l.dw, l.db = W.Data(), B.Data(), dW.Data(), dB.Data()
}

func (l *convCuda) Fprop(que Queue, in, work Array, trainMode bool) Array {
	if !SameShape(in.Dims(), l.InShape()) {
		panic(fmt.Errorf("fprop conv: invalid input shape: have %v, expect %v", in.Dims(), l.InShape()))
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

func (l *convCuda) Bprop(que Queue, grad, work Array) Array {
	if !SameShape(grad.Dims(), l.OutShape()) {
		panic(fmt.Errorf("bprop conv: invalid input shape: have %v, expect %v", grad.Dims(), l.OutShape()))
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
	if l.layerId > 0 {
		l.allocBprop(que)
		q.Call(
			args(C.CUDNN_EXECUTE+cuda.ConvBpropData, l.Algo[cuda.BwdDataAlgo], work.Size()*4, l.Ptr(), work.Data(),
				l.Filter.Ptr(), l.Dst.Ptr(), l.Src.Ptr(), l.w, grad.Data(), l.dSrc.Data()),
		)
	}
	return l.dSrc
}

// Create new max pooling layer, prev layer should be a ConvLayer
func NewPoolLayer(q Queue, inShape []int, size, stride int, pad, average bool) Layer {
	if len(inShape) != 4 {
		panic("PoolLayer: expect 4 dimensional input")
	}
	n, c, h, w := inShape[3], inShape[2], inShape[1], inShape[0]
	switch d := q.Dev().(type) {
	case cpuDevice:
		return newLayerMKL(mkl.Pooling(d.attr, n, c, h, w, size, stride, pad, average), true)
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
	l.allocBprop(q)
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

func (l *activationCuda) Fprop(q Queue, in, work Array, trainMode bool) Array {
	l.src = in
	q.Call(
		args(C.CUDNN_EXECUTE+cuda.ActivFprop, l.Ptr(), l.Src.Ptr(), in.Data(), l.dst.Data()),
	)
	return l.dst
}

func (l *activationCuda) Bprop(q Queue, grad, work Array) Array {
	l.allocBprop(q)
	q.Call(
		args(C.CUDNN_EXECUTE+cuda.ActivBprop, l.Ptr(), l.Src.Ptr(), l.dst.Data(), grad.Data(), l.src.Data(), l.dSrc.Data()),
	)
	return l.dSrc
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

func (a *activation) InShape() []int { return a.dst.Dims() }

func (a *activation) OutShape() []int { return a.dst.Dims() }

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
	a.allocBprop(q)
	if a.softmax {
		q.Call(Copy(grad, a.dSrc))
	} else {
		q.Call(a.bwd.setData(a.src, grad, a.dSrc))
	}
	return a.dSrc
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
	return 0, Bytes(l.dst, l.dSrc), l.Reserve.Size + l.States.Size
}

func (l *dropoutCuda) Release() {
	l.layerBase.Release()
	l.DropoutLayer.Release()
}

func (l *dropoutCuda) Fprop(q Queue, in, work Array, trainMode bool) Array {
	if !trainMode {
		return in
	}
	q.Call(
		args(C.CUDNN_EXECUTE+cuda.DropoutFprop, l.Ptr(), l.Src.Ptr(), in.Data(), l.dst.Data(), l.Reserve.Ptr, l.Reserve.Size),
	)
	return l.dst
}

func (l *dropoutCuda) Bprop(q Queue, grad, work Array) Array {
	l.allocBprop(q)
	q.Call(
		args(C.CUDNN_EXECUTE+cuda.DropoutBprop, l.Ptr(), l.Src.Ptr(), grad.Data(), l.dSrc.Data(), l.Reserve.Ptr, l.Reserve.Size),
	)
	return l.dSrc
}

type dropout struct {
	*layerBase
	ratio  float64
	filter Array
	mask   []float32
	rng    *rand.Rand
}

func (l *dropout) Memory() (weights, output, temp int) {
	return 0, Bytes(l.dst, l.dSrc), Bytes(l.filter) + 4*len(l.mask)
}

func (l *dropout) InShape() []int { return l.dst.Dims() }

func (l *dropout) OutShape() []int { return l.dst.Dims() }

func (l *dropout) Fprop(q Queue, in, work Array, trainMode bool) Array {
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

func (l *dropout) Bprop(q Queue, grad, work Array) Array {
	l.allocBprop(q)
	q.Call(
		Mul(l.filter, grad, l.dSrc),
	)
	return l.dSrc
}

// Create new batch normalisation layer
func NewBatchNormLayer(q Queue, avgFactor, epsilon float64, shape []int) BatchNormLayer {
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
	q.Call(Fill(p.runVar, 1))

	switch d := q.Dev().(type) {
	case cpuDevice:
		p.w = d.NewArray(Float32, c, 2)
		p.dw = d.NewArray(Float32, c, 2)
		q.Call(WriteCol(p.w, 0, ones(c)))
		l := &batchNormMkl{
			batchNorm: p,
			layerMKL:  newLayerMKL(mkl.BatchNorm(d.attr, n, c, h, w, epsilon), true),
		}
		l.SetStatsData(l.w.Data(), l.dw.Data(), p.mean.Data(), p.variance.Data())
		return l
	case gpuDevice:
		p.w = d.NewArray(Float32, c)
		p.b = d.NewArray(Float32, c)
		p.dw = d.NewArray(Float32, c)
		p.db = d.NewArray(Float32, c)
		q.Call(Fill(p.w, 1))
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
	w, b      Array
	dw, db    Array
	mean      Array
	variance  Array
	runMean   Array
	runVar    Array
}

func (l batchNorm) Stats() (w, b, runMean, runVar Array) {
	return l.w, l.b, l.runMean, l.runVar
}

func (l batchNorm) memory() (weights, temp int) {
	return Bytes(l.w, l.b, l.dw, l.db), Bytes(l.mean, l.variance, l.runMean, l.runVar)
}

func (l batchNorm) release() {
	for _, obj := range []Array{l.w, l.b, l.dw, l.db, l.mean, l.variance, l.runMean, l.runVar} {
		if obj != nil {
			obj.Release()
			obj = nil
		}
	}
}

type batchNormCuda struct {
	batchNorm
	*cuda.BatchNormLayer
	*layerBase
}

func (l *batchNormCuda) Release() {
	l.batchNorm.release()
	l.BatchNormLayer.Release()
	l.layerBase.Release()
}

func (l *batchNormCuda) Memory() (weights, output, temp int) {
	weights, temp = l.memory()
	return weights, Bytes(l.dst, l.dSrc), temp
}

func (l *batchNormCuda) Output() Array { return l.dst }

func (l *batchNormCuda) Fprop(q Queue, in, work Array, trainMode bool) Array {
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

func (l *batchNormCuda) Bprop(q Queue, grad, work Array) Array {
	l.allocBprop(q)
	q.Call(
		args(C.CUDNN_EXECUTE+cuda.BnormBprop, l.Src.Ptr(), l.src.Data(), grad.Data(), l.dSrc.Data(),
			l.Shape.Ptr(), l.w.Data(), l.dw.Data(), l.db.Data(), l.mean.Data(), l.variance.Data(),
			l.epsilon),
	)
	return l.dSrc
}

func (l *batchNormCuda) UpdateStats(q Queue, scale float32) {
	q.Call(
		Axpy(scale, l.dw, l.w),
		Axpy(scale, l.db, l.b),
	)
}

type batchNormMkl struct {
	batchNorm
	layerMKL
}

func (l *batchNormMkl) Release() {
	l.batchNorm.release()
	l.layerMKL.Release()
}

func (l *batchNormMkl) Memory() (weights, output, temp int) {
	weights, temp = l.memory()
	return weights, Bytes(l.dst, l.dSrc), temp
}

func (l *batchNormMkl) UpdateStats(q Queue, scale float32) {
	q.Call(
		Axpy(scale, l.dw, l.w),
		Axpy(1-l.avgFactor, l.runMean, l.runMean),
		Axpy(l.avgFactor, l.mean, l.runMean),
		Axpy(1-l.avgFactor, l.runVar, l.runVar),
		Axpy(l.avgFactor, l.variance, l.runVar),
	)
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

func (l layerMKL) Memory() (weights, output, temp int) {
	return 0, Bytes(l.dst, l.dSrc), l.Worksize() * 4
}

func (l layerMKL) SetParamData(W, B, dW, dB Array) {
	l.Layer.SetParams(W.Data(), B.Data(), dW.Data(), dB.Data())
}

func (l layerMKL) Output() Array { return l.dst }

func (l layerMKL) Fprop(q Queue, in, work Array, trainMode bool) Array {
	if !SameShape(in.Dims(), l.InShape()) {
		panic(fmt.Errorf("fprop: invalid input shape: have %v, expect %v", in.Dims(), l.InShape()))
	}
	l.SetSrc(in.Data())
	q.Call(dnnExecute(l.Fwd, l.ResPtr(), l.Type()+"_fprop"))
	return l.dst
}

func (l layerMKL) Bprop(q Queue, grad, work Array) Array {
	if !SameShape(grad.Dims(), l.OutShape()) {
		panic(fmt.Errorf("bprop: invalid input shape: have %v, expect %v", grad.Dims(), l.OutShape()))
	}
	l.SetDiffDst(grad.Data())
	if l.BBias != nil {
		q.Call(dnnExecute(l.BBias, l.ResPtr(), l.Type()+"_bprop_bias"))
	}
	if l.BFilter != nil {
		q.Call(dnnExecute(l.BFilter, l.ResPtr(), l.Type()+"_bprop_filter"))
	}
	if l.dSrc != nil {
		q.Call(dnnExecute(l.BData, l.ResPtr(), l.Type()+"_bprop_data"))
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

// base layer type
type layerBase struct {
	inShape []int
	src     Array
	dst     Array
	dSrc    Array
}

func newLayerBase(d Device, inShape, outShape []int) *layerBase {
	return &layerBase{
		inShape: inShape,
		dst:     d.NewArray(Float32, outShape...),
	}
}

func (l *layerBase) allocBprop(q Queue) {
	if l.dSrc == nil {
		l.dSrc = q.NewArray(Float32, l.inShape...)
	}
}

func (l *layerBase) Output() Array {
	return l.dst
}

func (l *layerBase) Memory() (weights, output, temp int) {
	return 0, Bytes(l.dst, l.dSrc), 0
}

func (l *layerBase) Release() {
	l.dst.Release()
	if l.dSrc != nil {
		l.dSrc.Release()
	}
}

func ones(n int) []float32 {
	arr := make([]float32, n)
	for i := range arr {
		arr[i] = 1
	}
	return arr
}
