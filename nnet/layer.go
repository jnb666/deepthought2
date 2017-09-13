package nnet

import (
	"encoding/json"
	"fmt"
	"github.com/jnb666/deepthought2/num"
	"github.com/jnb666/deepthought2/num/dnn"
	"math/rand"
)

// Layer interface type represents one layer of the neural net.
type Layer interface {
	Init(dev num.Device, inShape []int, prev Layer)
	Fprop(q num.Queue, input num.Array) num.Array
	Bprop(q num.Queue, grad num.Array) num.Array
	OutShape(inShape []int) []int
	String() string
}

// ParamLayer is a layer with weight and bias parameters
type ParamLayer interface {
	Layer
	InitParams(q num.Queue, scale float32, normal bool)
	Params() (W, B num.Array)
	ParamGrads() (dW, dB num.Array)
	SetParams(q num.Queue, W, B num.Array)
	UpdateParams(q num.Queue, learningRate, weightDecay float32)
}

// OutputLayer is the final layer in the stack
type OutputLayer interface {
	Layer
	Loss(q num.Queue, yOneHot, yPred num.Array) num.Array
}

// DNNLayer requires some extra initialisation to set inputs and outputs
type DNNLayer interface {
	Layer
	Get() dnn.Layer
	Link(dev num.Device, next Layer)
}

// Layer configuration details
type LayerConfig struct {
	Type string
	Data json.RawMessage
}

// Unmarshal JSON data and construct new layer
func (l LayerConfig) Unmarshal() Layer {
	switch l.Type {
	case "linear", "linearDNN":
		cfg := new(linearConfig)
		return cfg.Unmarshal(l.Type, l.Data)
	case "activation", "reluDNN":
		cfg := new(activationConfig)
		return cfg.Unmarshal(l.Type, l.Data)
	case "logRegression":
		cfg := new(logRegressionConfig)
		return cfg.Unmarshal(l.Type, l.Data)
	default:
		panic("invalid layer type: " + l.Type)
	}
}

func (l LayerConfig) String() string {
	return l.Unmarshal().String()
}

// Linear fully connected layer, implements ParamLayer interface.
func Linear(nout int) LayerConfig {
	data, _ := json.Marshal(linearConfig{Nout: nout})
	return LayerConfig{Type: "linear", Data: data}
}

// Sigmoid, tanh or relu activation layer, implements OutputLayer interface.
func Activation(typ string) LayerConfig {
	data, _ := json.Marshal(activationConfig{Atype: typ})
	return LayerConfig{Type: "activation", Data: data}
}

// LogRegression output layer with soft max activation.
func LogRegression() LayerConfig {
	return LayerConfig{Type: "logRegression"}
}

// DNN version of Linear fully connected layer.
func LinearDNN(nout int) LayerConfig {
	data, _ := json.Marshal(linearConfig{Nout: nout})
	return LayerConfig{Type: "linearDNN", Data: data}
}

// DNN version of relu activation layer.
func ReluDNN() LayerConfig {
	return LayerConfig{Type: "reluDNN"}
}

// weight and bias parameters
type paramBase struct {
	w, b   num.Array
	dw, db num.Array
}

func newParams(dev num.Device, wShape, bShape []int) paramBase {
	return paramBase{
		w:  dev.NewArray(num.Float32, wShape...),
		b:  dev.NewArray(num.Float32, bShape...),
		dw: dev.NewArray(num.Float32, wShape...),
		db: dev.NewArray(num.Float32, bShape...),
	}
}

func (p paramBase) Params() (W, B num.Array) {
	return p.w, p.b
}

func (p paramBase) ParamGrads() (dW, dB num.Array) {
	return p.dw, p.db
}

// linear layer implementation
type linearConfig struct {
	Nout int
}

func (c *linearConfig) Unmarshal(typ string, data json.RawMessage) Layer {
	if err := json.Unmarshal(data, c); err != nil {
		panic(err)
	}
	if typ == "linearDNN" {
		return &linearDNN{linearConfig: c}
	} else {
		return &linear{linearConfig: c}
	}
}

func (l *linearConfig) OutShape(inShape []int) []int {
	return []int{inShape[0], l.Nout}
}

type linear struct {
	*linearConfig
	paramBase
	last num.Array
	res  num.Array
	res2 num.Array
	ones num.Array
}

func (l *linear) String() string {
	if l.w != nil {
		d := l.w.Dims()
		return fmt.Sprintf("linear(%d,%d)", d[0], d[1])
	} else {
		return fmt.Sprintf("linear(%d)", l.Nout)
	}
}

func (l *linear) Init(dev num.Device, inShape []int, prev Layer) {
	if len(inShape) != 2 {
		panic("Linear: expect 2 dimensional input")
	}
	l.paramBase = newParams(dev, []int{inShape[1], l.Nout}, []int{l.Nout})
}

func (l *linear) Fprop(q num.Queue, input num.Array) num.Array {
	l.last = input
	l.res = allocArray(l.res, q, l.OutShape(input.Dims()))
	q.Call(
		num.Copy(l.res, l.b),
		num.Gemm(1, 1, input, l.w, l.res, num.NoTrans, num.NoTrans),
	)
	return l.res
}

func (l *linear) Bprop(q num.Queue, grad num.Array) num.Array {
	scale := 1 / float32(grad.Dims()[0])
	l.res2 = allocArray(l.res2, q, l.last.Dims())
	l.ones = allocOnes(l.ones, q, l.last.Dims()[0])
	q.Call(
		num.Gemv(scale, 0, grad, l.ones, l.db, num.Trans),
		num.Gemm(scale, 0, l.last, grad, l.dw, num.Trans, num.NoTrans),
		num.Gemm(1, 0, grad, l.w, l.res2, num.NoTrans, num.Trans),
	)
	return l.res2
}

func (l *linear) InitParams(q num.Queue, scale float32, normal bool) {
	q.Call(
		num.Write(l.w, randomWeights(l.w.Dims(), scale, normal)),
	)
}

func (l *linear) SetParams(q num.Queue, W, B num.Array) {
	q.Call(
		num.Copy(l.w, W),
		num.Copy(l.b, B),
	)
}

func (l *linear) UpdateParams(q num.Queue, learningRate, weightDecay float32) {
	if weightDecay != 0 {
		q.Call(num.Axpy(-weightDecay, l.w, l.dw))
	}
	q.Call(
		num.Axpy(-learningRate, l.dw, l.w),
		num.Axpy(-learningRate, l.db, l.b),
	)
}

type linearDNN struct {
	*linearConfig
	paramBase
	inShape []int
	layer   dnn.Layer
	res     num.Array
	srcConv bool
	dstConv bool
}

func (l *linearDNN) Get() dnn.Layer { return l.layer }

func (l *linearDNN) String() string {
	if l.w != nil {
		d := l.w.Dims()
		return fmt.Sprintf("linearDNN(%d,%d)", d[0], d[1])
	} else {
		return fmt.Sprintf("linearDNN(%d)", l.Nout)
	}
}

func (l *linearDNN) Init(dev num.Device, inShape []int, prev Layer) {
	if len(inShape) != 2 {
		panic("LinearDNN: expect 2 dimensional input")
	}
	l.inShape = inShape
	l.paramBase = newParams(dev, []int{inShape[1], l.Nout}, []int{l.Nout})
	l.layer = dev.LinearLayer(inShape[0], inShape[1], l.Nout)
	num.ArrayIn(l.layer, dnn.Filter, l.w, num.NoTrans)
	num.ArrayIn(l.layer, dnn.Bias, l.b, num.NoTrans)
	num.ArrayIn(l.layer, dnn.DiffFilter, l.dw, num.NoTrans)
	num.ArrayOut(l.layer, dnn.DiffFilter, l.dw, num.NoTrans)
	num.ArrayOut(l.layer, dnn.DiffBias, l.db, num.NoTrans)
	var p DNNLayer
	if prev != nil {
		p, _ = prev.(DNNLayer)
	}
	if p == nil {
		l.srcConv = num.ConvIn(l.layer, dnn.Src, inShape, num.Trans)
	} else {
		l.layer.SetData(dnn.Src, p.Get().Data(dnn.Dst))
	}
}

func (l *linearDNN) Link(dev num.Device, next Layer) {
	var n DNNLayer
	if next != nil {
		n, _ = next.(DNNLayer)
	}
	if n == nil {
		outShape := l.OutShape(l.inShape)
		l.res = dev.NewArray(num.Float32, outShape...)
		l.dstConv = num.ArrayOut(l.layer, dnn.Dst, l.res, num.Trans)
		num.ConvIn(l.layer, dnn.DiffDst, outShape, num.Trans)
	} else {
		l.layer.SetData(dnn.DiffDst, n.Get().Data(dnn.DiffSrc))
	}
}

func (l *linearDNN) Fprop(q num.Queue, input num.Array) num.Array {
	if l.srcConv {
		q.Call(num.Set(l.layer, dnn.Src, input))
	}
	q.Call(num.Fprop(l.layer))
	if l.dstConv {
		q.Call(num.Get(l.layer, dnn.Dst, l.res))
	}
	return l.res
}

func (l *linearDNN) Bprop(q num.Queue, grad num.Array) num.Array {
	if l.dstConv {
		q.Call(num.Set(l.layer, dnn.DiffDst, grad))
	}
	scale := 1 / float32(l.layer.InShape()[1])
	q.Call(
		num.BpropData(l.layer),
		num.BpropFilter(l.layer),
		num.BpropBias(l.layer),
		num.Get(l.layer, dnn.DiffFilter, l.dw),
		num.Get(l.layer, dnn.DiffBias, l.db),
		num.Scale(scale, l.dw),
		num.Scale(scale, l.db),
	)
	return nil
}

func (l *linearDNN) InitParams(q num.Queue, scale float32, normal bool) {
	q.Call(
		num.Write(l.w, randomWeights(l.w.Dims(), scale, normal)),
		num.Set(l.layer, dnn.Filter, l.w),
		num.Set(l.layer, dnn.Bias, l.b),
	)
}

func (l *linearDNN) SetParams(q num.Queue, W, B num.Array) {
	q.Call(
		num.Copy(l.w, W),
		num.Copy(l.b, B),
		num.Set(l.layer, dnn.Filter, l.w),
		num.Set(l.layer, dnn.Bias, l.b),
	)
}

func (l *linearDNN) UpdateParams(q num.Queue, learningRate, weightDecay float32) {
	if weightDecay != 0 {
		q.Call(
			num.Axpy(-weightDecay, l.w, l.dw),
			num.Set(l.layer, dnn.DiffFilter, l.dw),
		)
	}
	q.Call(
		num.Axpy(-learningRate, l.dw, l.w),
		num.Axpy(-learningRate, l.db, l.b),
		num.Set(l.layer, dnn.Filter, l.w),
		num.Set(l.layer, dnn.Bias, l.b),
	)
}

type activationConfig struct {
	Atype string
}

func (c *activationConfig) Unmarshal(typ string, data json.RawMessage) Layer {
	if typ == "reluDNN" {
		return &reluDNN{}
	}
	if err := json.Unmarshal(data, c); err != nil {
		panic(err)
	}
	layer := &activation{activationConfig: c}
	switch c.Atype {
	case "sigmoid":
		layer.activ = num.Sigmoid
		layer.deriv = num.SigmoidD
	case "tanh":
		layer.activ = num.Tanh
		layer.deriv = num.TanhD
	case "relu":
		layer.activ = num.Relu
		layer.deriv = num.ReluD
	default:
		panic(fmt.Sprintf("activation type %s invalid", c.Atype))
	}
	return layer
}

type activation struct {
	*activationConfig
	activ func(x, y num.Array) num.Function
	deriv func(x, y, z num.Array) num.Function
	last  num.Array
	res   num.Array
	res2  num.Array
}

func (l *activation) Init(dev num.Device, inShape []int, prev Layer) {}

func (l *activation) OutShape(inShape []int) []int { return inShape }

func (l *activation) String() string {
	return fmt.Sprintf("activation(%s)", l.Atype)
}

func (l *activation) Fprop(q num.Queue, input num.Array) num.Array {
	l.last = input
	l.res = allocArray(l.res, q, input.Dims())
	q.Call(l.activ(input, l.res))
	return l.res
}

func (l *activation) Loss(q num.Queue, yOneHot, yPred num.Array) num.Array {
	l.res2 = allocArray(l.res2, q, yPred.Dims())
	q.Call(num.QuadraticLoss(yOneHot, yPred, l.res2))
	return l.res2
}

func (l *activation) Bprop(q num.Queue, grad num.Array) num.Array {
	q.Call(l.deriv(l.last, grad, l.res))
	return l.res
}

// reluDNN layer must always be preceded by another DNN layer such as linearDNN
type reluDNN struct {
	layer   dnn.Layer
	shape   []int
	res     num.Array
	dstConv bool
}

func (l *reluDNN) OutShape(inShape []int) []int { return inShape }

func (l *reluDNN) String() string { return fmt.Sprintf("reluDNN") }

func (l *reluDNN) Get() dnn.Layer { return l.layer }

func (l *reluDNN) Init(dev num.Device, inShape []int, prev Layer) {
	l.shape = inShape
	l.layer = dev.ReluLayer(prev.(DNNLayer).Get())
}

func (l *reluDNN) Link(dev num.Device, next Layer) {
	var n DNNLayer
	if next != nil {
		n, _ = next.(DNNLayer)
	}
	if n == nil {
		l.res = dev.NewArray(num.Float32, l.shape...)
		l.dstConv = num.ArrayOut(l.layer, dnn.Dst, l.res, num.Trans)
		num.ConvIn(l.layer, dnn.DiffDst, l.shape, num.Trans)
	} else {
		l.layer.SetData(dnn.DiffDst, n.Get().Data(dnn.DiffSrc))
	}
}

func (l *reluDNN) Fprop(q num.Queue, input num.Array) num.Array {
	q.Call(num.Fprop(l.layer))
	if l.dstConv {
		q.Call(num.Get(l.layer, dnn.Dst, l.res))
	}
	return l.res
}

func (l *reluDNN) Bprop(q num.Queue, grad num.Array) num.Array {
	if l.dstConv {
		q.Call(num.Set(l.layer, dnn.DiffDst, grad))
	}
	q.Call(num.BpropData(l.layer))
	return nil
}

type logRegressionConfig struct{}

func (c *logRegressionConfig) Unmarshal(typ string, data json.RawMessage) Layer {
	return &logRegression{}
}

type logRegression struct {
	*logRegressionConfig
	res  num.Array
	res2 num.Array
}

func (l *logRegression) Init(dev num.Device, inShape []int, prev Layer) {}

func (l *logRegression) String() string { return "logRegression" }

func (l *logRegression) OutShape(inShape []int) []int { return inShape }

func (l *logRegression) Fprop(q num.Queue, input num.Array) num.Array {
	l.res = allocArray(l.res, q, input.Dims())
	q.Call(num.Softmax(input, l.res))
	return l.res
}

func (l *logRegression) Loss(q num.Queue, yOneHot, yPred num.Array) num.Array {
	l.res2 = allocArray(l.res2, q, yPred.Dims())
	q.Call(num.SoftmaxLoss(yOneHot, yPred, l.res2))
	return l.res2
}

func (l *logRegression) Bprop(q num.Queue, grad num.Array) num.Array {
	q.Call(num.Copy(l.res, grad))
	return l.res
}

// utilities
func randomWeights(dims []int, scale float32, normal bool) []float32 {
	weights := make([]float32, num.Prod(dims))
	for i := range weights {
		if normal {
			weights[i] = float32(rand.NormFloat64()) * scale
		} else {
			weights[i] = rand.Float32() * scale
		}
	}
	return weights
}

func allocArray(a num.Array, q num.Queue, shape []int) num.Array {
	if a == nil || !num.SameShape(a.Dims(), shape) {
		a = q.NewArray(num.Float32, shape...)
	}
	return a
}

func allocOnes(a num.Array, q num.Queue, size int) num.Array {
	if a == nil || a.Dims()[0] != size {
		a = q.NewArray(num.Float32, size)
		q.Call(num.Fill(a, 1))
	}
	return a
}
