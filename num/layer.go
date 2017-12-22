package num

/*
#include "num.h"
*/
import "C"

import (
	"fmt"
	"github.com/jnb666/deepthought2/num/cuda"
	"github.com/jnb666/deepthought2/num/mkl"
	"unsafe"
)

// Layer interface type represents an Activation or MaxPool layer
type Layer interface {
	InShape() []int
	OutShape() []int
	Fprop(q Queue, in, work Array) Array
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
func ConvLayer(q Queue, ix int, inShape []int, nFeats, size, stride, pad int) (ParamLayer, int) {
	if len(inShape) != 4 {
		panic("ConvLayer: expect 4 dimensional input")
	}
	n, c, h, w := inShape[3], inShape[2], inShape[1], inShape[0]
	switch d := q.Dev().(type) {
	case cpuDevice:
		return newLayerMKL(mkl.Convolution(d.attr, n, c, h, w, nFeats, size, stride, pad), ix != 0), 0
	case gpuDevice:
		layer := cuda.Convolution(n, c, h, w, nFeats, size, stride, pad)
		workSize := layer.Init(q.(*gpuQueue).stream)
		l := &convCuda{
			ConvLayer: layer,
			dst:       d.NewArray(Float32, layer.OutShape()...),
		}
		if ix != 0 {
			l.diffSrc = d.NewArray(Float32, layer.InShape()...)
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
	diffSrc  Array
}

func (l *convCuda) Release() {
	l.dst.Release()
	if l.diffSrc != nil {
		l.diffSrc.Release()
	}
	l.ConvLayer.Release()
}

func (l *convCuda) SetParamData(W, B, dW, dB Array) {
	l.w, l.b, l.dw, l.db = W.Data(), B.Data(), dW.Data(), dB.Data()
}

func (l *convCuda) Output() Array { return l.dst }

func (l *convCuda) Fprop(que Queue, in, work Array) Array {
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
	if l.diffSrc != nil {
		q.Call(
			args(C.CUDNN_EXECUTE+cuda.ConvBpropData, l.Algo[cuda.BwdDataAlgo], work.Size(), l.Ptr(), work.Data(),
				l.Filter.Ptr(), l.Dst.Ptr(), l.Src.Ptr(), l.w, grad.Data(), l.diffSrc.Data()),
		)
	}
	return l.diffSrc
}

// Create new max pooling layer, prev layer should be a ConvLayer
func MaxPoolLayer(q Queue, inShape []int, size, stride int) Layer {
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
			diffSrc:   d.NewArray(Float32, layer.InShape()...),
		}
	default:
		panic("device type not supported")
	}
}

type poolCuda struct {
	*cuda.PoolLayer
	src, dst Array
	diffSrc  Array
}

func (l *poolCuda) Release() {
	l.dst.Release()
	l.diffSrc.Release()
	l.PoolLayer.Release()
}

func (l *poolCuda) Output() Array { return l.dst }

func (l *poolCuda) Fprop(q Queue, in, work Array) Array {
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
		args(C.CUDNN_EXECUTE+cuda.PoolBprop, l.Ptr(), l.Src.Ptr(), l.Dst.Ptr(), l.dst.Data(), grad.Data(), l.src.Data(), l.diffSrc.Data()),
	)
	return l.diffSrc
}

// Create new activation layer, typ may be sigmoid, tanh or relu
func ActivationLayer(q Queue, typ string, shape []int) Layer {
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
			dst:        d.NewArray(Float32, layer.OutShape()...),
			diffSrc:    d.NewArray(Float32, layer.InShape()...),
		}
	default:
		panic("device type not supported")
	}
}

type activationCuda struct {
	*cuda.ActivLayer
	src, dst Array
	diffSrc  Array
}

func (l *activationCuda) Release() {
	l.dst.Release()
	l.diffSrc.Release()
	l.ActivLayer.Release()
}

func (l *activationCuda) Output() Array { return l.dst }

func (l *activationCuda) Fprop(q Queue, in, work Array) Array {
	l.src = in
	q.Call(
		args(C.CUDNN_EXECUTE+cuda.ActivFprop, l.Ptr(), l.Src.Ptr(), in.Data(), l.dst.Data()),
	)
	return l.dst
}

func (l *activationCuda) Bprop(q Queue, grad, work Array) Array {
	q.Call(
		args(C.CUDNN_EXECUTE+cuda.ActivBprop, l.Ptr(), l.Src.Ptr(), l.dst.Data(), grad.Data(), l.src.Data(), l.diffSrc.Data()),
	)
	return l.diffSrc
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

func (a *activation) Fprop(q Queue, in, work Array) Array {
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

// layer which wraps Intel MKL DNN layer
type layerMKL struct {
	*mkl.Layer
	dst, diffSrc Array
}

func newLayerMKL(layer *mkl.Layer, bpropData bool) layerMKL {
	l := layerMKL{
		Layer: layer,
		dst:   newArrayCPU(Float32, layer.OutShape(), layer.Dst()),
	}
	if bpropData {
		l.diffSrc = newArrayCPU(Float32, layer.InShape(), layer.DiffSrc())
	}
	return l
}

func (l layerMKL) Release() {}

func (l layerMKL) SetParamData(W, B, dW, dB Array) {
	l.Layer.SetParams(W.Data(), B.Data(), dW.Data(), dB.Data())
}

func (l layerMKL) Output() Array { return l.dst }

func (l layerMKL) Fprop(q Queue, in, work Array) Array {
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
	if l.diffSrc != nil {
		q.Call(
			dnnExecute(l.BData, l.ResPtr(), l.Type()+"_bprop_data"),
		)
	}
	return l.diffSrc
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
