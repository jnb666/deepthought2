package num

import (
	"github.com/jnb666/deepthought2/num/mkl"
)

// Layer interface type represents a DNN layer
type Layer interface {
	Dst() Array
	DiffSrc() Array
	SetSrc(Array)
	SetDiffDst(Array)
	SetParams(W, B, dW, dB Array)
	HasParams() bool
	Type() string
	InShape() []int
	OutShape() []int
	FilterShape() []int
	BiasShape() []int
}

type layerCPU struct {
	*mkl.Layer
}

// Create new layers
func (d cpuDevice) LinearLayer(nBatch, nIn, nOut int) Layer {
	return layerCPU{mkl.InnerProduct(d.attr, nBatch, nIn, nOut)}
}

func (d cpuDevice) ConvLayer(nBatch, depth, h, w, nFeats, size, stride, pad int) Layer {
	return layerCPU{mkl.Convolution(d.attr, nBatch, depth, h, w, nFeats, size, stride, pad)}
}

func (d cpuDevice) MaxPoolLayer(prev Layer, size, stride int) Layer {
	return layerCPU{mkl.MaxPooling(d.attr, prev.(layerCPU).Layer, size, stride)}
}

func (d cpuDevice) ReluLayer(prev Layer) Layer {
	return layerCPU{mkl.Relu(d.attr, prev.(layerCPU).Layer)}
}

// Set layer parameters
func (l layerCPU) Dst() Array {
	return &arrayCPU{dims: l.OutShape(), dtype: Float32, data: l.Layer.Dst()}
}

func (l layerCPU) DiffSrc() Array {
	return &arrayCPU{dims: l.InShape(), dtype: Float32, data: l.Layer.DiffSrc()}
}

func (l layerCPU) SetSrc(a Array) {
	l.Layer.SetSrc(a.(*arrayCPU).data)
}

func (l layerCPU) SetDiffDst(a Array) {
	l.Layer.SetDiffDst(a.(*arrayCPU).data)
}

func (l layerCPU) SetParams(W, B, dW, dB Array) {
	l.Layer.SetParams(W.(*arrayCPU).data, B.(*arrayCPU).data, dW.(*arrayCPU).data, dB.(*arrayCPU).data)
}

// Forward propagation
func Fprop(layer Layer) Function {
	l := layer.(layerCPU).Layer
	return dnnExecute(l.Fwd, l.ResPtr(), l.Type()+"_fprop")
}

// Backward propagation
func BpropData(layer Layer) Function {
	l := layer.(layerCPU).Layer
	return dnnExecute(l.BData, l.ResPtr(), l.Type()+"_bprop")
}

func BpropFilter(layer Layer) Function {
	l := layer.(layerCPU).Layer
	return dnnExecute(l.BFilter, l.ResPtr(), l.Type()+"_bprop")
}

func BpropBias(layer Layer) Function {
	l := layer.(layerCPU).Layer
	return dnnExecute(l.BBias, l.ResPtr(), l.Type()+"_bprop")
}
