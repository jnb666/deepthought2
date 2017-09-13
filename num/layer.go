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

func (d cpuDevice) ReluLayer(prev dnn.Layer) dnn.Layer {
	out := mkl.NewLayout(prev.OutShape(), dnn.ColMajor)
	return mkl.Relu(d.attr, prev.(*mkl.Layer), out)
}

// Setup conversion from array to layer resource
func ConvIn(layer dnn.Layer, typ dnn.ResType, dims []int, trans TransType) bool {
	l := layer.(*mkl.Layer)
	if l.InitInConv(typ, dims, trans2Layout[trans]) {
		l.Alloc(typ)
		return true
	}
	return false
}

func ArrayIn(layer dnn.Layer, typ dnn.ResType, a Array, trans TransType) bool {
	if ConvIn(layer, typ, a.Dims(), trans) {
		return true
	}
	layer.SetData(typ, a.Data())
	return false
}

// Setup conversion from layer resource to array
func ConvOut(layer dnn.Layer, typ dnn.ResType, dims []int, trans TransType) bool {
	l := layer.(*mkl.Layer)
	if l.InitOutConv(typ, dims, trans2Layout[trans]) {
		l.Alloc(typ)
		return true
	}
	return false
}

func ArrayOut(layer dnn.Layer, typ dnn.ResType, a Array, trans TransType) bool {
	if ConvOut(layer, typ, a.Dims(), trans) {
		return true
	}
	a.SetData(layer.Data(typ))
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
	return dnnExecute(l.Fwd, l.ResPtr())
}

// Backward propagation
func BpropData(layer dnn.Layer) Function {
	l := layer.(*mkl.Layer)
	return dnnExecute(l.BData, l.ResPtr())
}

func BpropFilter(layer dnn.Layer) Function {
	l := layer.(*mkl.Layer)
	return dnnExecute(l.BFilter, l.ResPtr())
}

func BpropBias(layer dnn.Layer) Function {
	l := layer.(*mkl.Layer)
	return dnnExecute(l.BBias, l.ResPtr())
}
