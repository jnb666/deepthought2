package num

import (
	"github.com/jnb666/deepthought2/num/dnn"
	"github.com/jnb666/deepthought2/num/mkl"
)

var trans2Layout = map[TransType]dnn.DataLayout{
	NoTrans: dnn.ColMajor,
	Trans:   dnn.RowMajor,
}

// Create new layers
func (d cpuDevice) LinearLayer(nBatch, nIn, nOut int) dnn.Layer {
	return mkl.InnerProduct(d.attr, nBatch, nIn, nOut, dnn.ColMajor)
}

func (d cpuDevice) ConvLayer(nBatch, depth, h, w, nFeats, size, stride, pad int) dnn.Layer {
	return mkl.Convolution(d.attr, nBatch, depth, h, w, nFeats, size, stride, pad, dnn.ColMajor)
}

func (d cpuDevice) MaxPoolLayer(prev dnn.Layer, size, stride int) dnn.Layer {
	return mkl.MaxPooling(d.attr, prev.(*mkl.Layer), size, stride)
}

func (d cpuDevice) ReluLayer(prev dnn.Layer) dnn.Layer {
	return mkl.Relu(d.attr, prev.(*mkl.Layer))
}

// Set layer parameters
func SetParams(layer dnn.Layer, W, B, dW, dB Array) {
	layer.SetData(dnn.Filter, W.Data())
	layer.SetData(dnn.Bias, B.Data())
	layer.SetData(dnn.DiffFilter, dW.Data())
	layer.SetData(dnn.DiffBias, dB.Data())
}

// Setup conversion from array to layer resource
func In(layer dnn.Layer, typ dnn.ResType, dims []int, trans TransType) bool {
	l := layer.(*mkl.Layer)
	if l.InitInConv(typ, dims, trans2Layout[trans]) {
		l.Alloc(typ)
		return true
	}
	return false
}

// Setup conversion from layer resource to array
func Out(layer dnn.Layer, typ dnn.ResType, dims []int, trans TransType) bool {
	l := layer.(*mkl.Layer)
	if l.InitOutConv(typ, dims, trans2Layout[trans]) {
		l.Alloc(typ)
		return true
	}
	return false
}

// Set Layer attribute from array
func Set(layer dnn.Layer, typ dnn.ResType, a Array) Function {
	l := layer.(*mkl.Layer)
	return dnnConvert(l.InConv(typ), a.Data(), l.Data(typ))
}

// Get layer attribute to array
func Get(layer dnn.Layer, typ dnn.ResType, a Array) Function {
	l := layer.(*mkl.Layer)
	return dnnConvert(l.OutConv(typ), l.Data(typ), a.Data())
}

// Forward propagation
func Fprop(layer dnn.Layer) Function {
	l := layer.(*mkl.Layer)
	return dnnExecute(l.Fwd, l.ResPtr(), l.Type()+"_fprop")
}

// Backward propagation
func BpropData(layer dnn.Layer) Function {
	l := layer.(*mkl.Layer)
	return dnnExecute(l.BData, l.ResPtr(), l.Type()+"_bprop")
}

func BpropFilter(layer dnn.Layer) Function {
	l := layer.(*mkl.Layer)
	return dnnExecute(l.BFilter, l.ResPtr(), l.Type()+"_bprop")
}

func BpropBias(layer dnn.Layer) Function {
	l := layer.(*mkl.Layer)
	return dnnExecute(l.BBias, l.ResPtr(), l.Type()+"_bprop")
}
