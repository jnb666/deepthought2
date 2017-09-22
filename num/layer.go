package num

/*
#include "num.h"
*/
import "C"

import (
	"github.com/jnb666/deepthought2/num/mkl"
)

// Layer interface type represents a DNN layer
type Layer interface {
	InShape() []int
	OutShape() []int
	FilterShape() []int
	BiasShape() []int
	Fprop(q Queue, in Array) Array
	Bprop(q Queue, grad Array) Array
	SetParamData(W, B, dW, dB Array)
	DNNLayer() Layer
}

// Create new convolution layer, input shape is nBatch x depth x h x w
func ConvLayer(dev Device, nBatch, depth, h, w, nFeats, size, stride, pad int) Layer {
	return newLayerMKL(mkl.Convolution(dev.(cpuDevice).attr, nBatch, depth, h, w, nFeats, size, stride, pad))
}

// Create new max pooling layer, prev layer should be a ConvLayer
func MaxPoolLayer(dev Device, prev Layer, size, stride int) Layer {
	return newLayerMKL(mkl.MaxPooling(dev.(cpuDevice).attr, prev.(layerMKL).Layer, size, stride))
}

// Create new activation layer, typ may be sigmoid, tanh or relu
func ActivationLayer(dev Device, typ string, shape []int, prev Layer) Layer {
	if prev != nil {
		if typ == "relu" {
			return newLayerMKL(mkl.Relu(dev.(cpuDevice).attr, prev.(layerMKL).Layer))
		}
	} else {
		switch typ {
		case "sigmoid":
			return newActivation(dev, C.SIGMOID, C.SIGMOID_D, shape)
		case "tanh":
			return newActivation(dev, C.TANH, C.TANH_D, shape)
		case "relu":
			return newActivation(dev, C.RELU, C.RELU_D, shape)
		}
	}
	panic("ActivationLayer: type " + typ + " not supported")
}

type activation struct {
	fwd, bwd       Function
	src, dst, dsrc Array
}

func newActivation(dev Device, fwd, bwd int, shape []int) *activation {
	size := Prod(shape)
	return &activation{
		fwd:  args(fwd, size),
		bwd:  args(bwd, size),
		dst:  dev.NewArray(Float32, shape...),
		dsrc: dev.NewArray(Float32, shape...),
	}
}

func (a *activation) Fprop(q Queue, in Array) Array {
	a.src = in
	q.Call(a.fwd.setData(a.src, a.dst))
	return a.dst
}

func (a *activation) Bprop(q Queue, grad Array) Array {
	q.Call(a.bwd.setData(a.src, grad, a.dsrc))
	return a.dsrc
}

func (a *activation) InShape() []int { return a.dst.Dims() }

func (a *activation) OutShape() []int { return a.dst.Dims() }

func (a *activation) FilterShape() []int { return nil }

func (a *activation) BiasShape() []int { return nil }

func (a *activation) DNNLayer() Layer { return nil }

func (a *activation) SetParamData(W, B, dW, dB Array) { panic("no params for activation layer") }

// layer which wraps Intel MKL DNN layer
type layerMKL struct {
	*mkl.Layer
	dst, diffSrc Array
}

func newLayerMKL(l *mkl.Layer) layerMKL {
	return layerMKL{
		Layer:   l,
		dst:     &arrayCPU{dims: l.OutShape(), dtype: Float32, data: l.Dst()},
		diffSrc: &arrayCPU{dims: l.InShape(), dtype: Float32, data: l.DiffSrc()},
	}
}

func (l layerMKL) DNNLayer() Layer { return l }

func (l layerMKL) SetParamData(W, B, dW, dB Array) {
	l.Layer.SetParams(W.(*arrayCPU).data, B.(*arrayCPU).data, dW.(*arrayCPU).data, dB.(*arrayCPU).data)
}

func (l layerMKL) Fprop(q Queue, in Array) Array {
	l.SetSrc(in.Data())
	q.Call(
		dnnExecute(l.Fwd, l.ResPtr(), l.Type()+"_fprop"),
	)
	return l.dst
}

func (l layerMKL) Bprop(q Queue, grad Array) Array {
	l.SetDiffDst(grad.Data())
	q.Call(
		dnnExecute(l.BData, l.ResPtr(), l.Type()+"_bprop"),
	)
	if l.HasParams() {
		q.Call(
			dnnExecute(l.BFilter, l.ResPtr(), l.Type()+"_bprop"),
			dnnExecute(l.BBias, l.ResPtr(), l.Type()+"_bprop"),
		)
	}
	return l.diffSrc
}
