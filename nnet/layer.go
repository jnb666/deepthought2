package nnet

import (
	"encoding/json"
	"fmt"
	"github.com/jnb666/deepthought2/num"
	"math/rand"
)

const fixedWeights = false

// Layer interface type represents one layer of the neural net.
type Layer interface {
	Fprop(q num.Queue, input num.Array) num.Array
	Bprop(q num.Queue, grad num.Array) num.Array
	OutShape(inShape []int) []int
	String() string
}

// ParamLayer is a layer with weight and bias parameters
type ParamLayer interface {
	Layer
	InitParams(q num.Queue, inShape []int, scale float32, normal bool)
	Params() (W, B num.Array)
	ParamGrads() (dW, dB num.Array)
}

// OutputLayer is the final layer in the stack
type OutputLayer interface {
	Layer
	Loss(q num.Queue, yOneHot, yPred num.Array) num.Array
}

// Layer configuration details
type LayerConfig struct {
	Type string
	Data json.RawMessage
}

// Unmarshal JSON data and construct new layer
func (l LayerConfig) Unmarshal() Layer {
	switch l.Type {
	case "linear":
		var cfg = new(linearConfig)
		return cfg.Unmarshal(l.Data)
	case "activation":
		var cfg = new(activationConfig)
		return cfg.Unmarshal(l.Data)
	case "logRegression":
		var cfg = new(logRegressionConfig)
		return cfg.Unmarshal(l.Data)
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

type linearConfig struct {
	Nout int
}

func (c *linearConfig) Unmarshal(data json.RawMessage) Layer {
	if err := json.Unmarshal(data, c); err != nil {
		panic(err)
	}
	return &linear{linearConfig: *c}
}

type linear struct {
	linearConfig
	w, b   num.Array
	dw, db num.Array
	last   num.Array
	res    num.Array
	res2   num.Array
	ones   num.Array
}

func (l *linear) String() string {
	if l.w != nil {
		d := l.w.Dims()
		return fmt.Sprintf("linear(%d,%d)", d[0], d[1])
	} else {
		return fmt.Sprintf("linear(%d)", l.Nout)
	}
}

func (l *linear) InitParams(q num.Queue, inShape []int, scale float32, normal bool) {
	l.w = num.NewArray(q.Device(), num.Float32, inShape[0], l.Nout)
	l.dw = num.NewArray(q.Device(), num.Float32, inShape[0], l.Nout)
	l.b = num.NewArray(q.Device(), num.Float32, l.Nout)
	l.db = num.NewArray(q.Device(), num.Float32, l.Nout)
	weights := make([]float32, inShape[0]*l.Nout)
	if fixedWeights {
		w := 0
		for row := 0; row < inShape[0]; row++ {
			for col := 0; col < l.Nout; col++ {
				w = (w + 23) % 19
				weights[row+col*inShape[0]] = float32(w) / 25
			}
		}
	} else {
		for i := range weights {
			if normal {
				weights[i] = float32(rand.NormFloat64()) * scale
			} else {
				weights[i] = rand.Float32() * scale
			}
		}
	}
	q.Call(num.Write(l.w, weights))
}

func (l *linear) Params() (W, B num.Array) {
	return l.w, l.b
}

func (l *linear) ParamGrads() (dW, dB num.Array) {
	return l.dw, l.db
}

func (l *linear) OutShape(inShape []int) []int {
	if len(inShape) == 1 {
		return []int{l.Nout}
	} else {
		return []int{inShape[0], l.Nout}
	}
}

func (l *linear) Fprop(q num.Queue, input num.Array) num.Array {
	l.last = input
	l.res = allocArray(l.res, q.Device(), l.OutShape(input.Dims()))
	q.Call(
		num.Copy(l.res, l.b),
		num.Gemm(1, 1, input, l.w, l.res, num.NoTrans, num.NoTrans),
	)
	return l.res
}

func (l *linear) Bprop(q num.Queue, grad num.Array) num.Array {
	scale := 1 / float32(grad.Dims()[0])
	l.res2 = allocArray(l.res2, q.Device(), l.last.Dims())
	l.ones = allocOnes(l.ones, q, l.last.Dims()[0])
	q.Call(
		num.Gemv(scale, 0, grad, l.ones, l.db, num.Trans),
		num.Gemm(scale, 0, l.last, grad, l.dw, num.Trans, num.NoTrans),
		num.Gemm(1, 0, grad, l.w, l.res2, num.NoTrans, num.Trans),
	)
	return l.res2
}

// Sigmoid activation layer, implements OutputLayer interface.
func Activation(typ string) LayerConfig {
	data, _ := json.Marshal(activationConfig{Atype: typ})
	return LayerConfig{Type: "activation", Data: data}
}

type activationConfig struct {
	Atype string
}

func (c *activationConfig) Unmarshal(data json.RawMessage) Layer {
	if err := json.Unmarshal(data, c); err != nil {
		panic(err)
	}
	layer := &activation{activationConfig: *c}
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
	activationConfig
	activ func(x, y num.Array) num.Function
	deriv func(x, y, z num.Array) num.Function
	last  num.Array
	res   num.Array
	res2  num.Array
}

func (l activation) String() string {
	return fmt.Sprintf("activation(%s)", l.Atype)
}

func (l *activation) OutShape(inShape []int) []int {
	return inShape
}

func (l *activation) Fprop(q num.Queue, input num.Array) num.Array {
	l.last = input
	l.res = allocArray(l.res, q.Device(), input.Dims())
	q.Call(l.activ(input, l.res))
	return l.res
}

func (l *activation) Loss(q num.Queue, yOneHot, yPred num.Array) num.Array {
	l.res2 = allocArray(l.res2, q.Device(), yPred.Dims())
	q.Call(num.QuadraticLoss(yOneHot, yPred, l.res2))
	return l.res2
}

func (l *activation) Bprop(q num.Queue, grad num.Array) num.Array {
	q.Call(l.deriv(l.last, grad, l.res))
	return l.res
}

// LogRegrssion output layer with soft max activation.
func LogRegression() LayerConfig {
	return LayerConfig{Type: "logRegression"}
}

type logRegressionConfig struct{}

func (c *logRegressionConfig) Unmarshal(data json.RawMessage) Layer {
	return &logRegression{}
}

type logRegression struct {
	logRegressionConfig
	res  num.Array
	res2 num.Array
}

func (l logRegression) String() string {
	return "logRegression"
}

func (l *logRegression) OutShape(inShape []int) []int {
	return inShape
}

func (l *logRegression) Fprop(q num.Queue, input num.Array) num.Array {
	l.res = allocArray(l.res, q.Device(), input.Dims())
	q.Call(num.Softmax(input, l.res))
	return l.res
}

func (l *logRegression) Loss(q num.Queue, yOneHot, yPred num.Array) num.Array {
	l.res2 = allocArray(l.res2, q.Device(), yPred.Dims())
	q.Call(num.SoftmaxLoss(yOneHot, yPred, l.res2))
	return l.res2
}

func (l *logRegression) Bprop(q num.Queue, grad num.Array) num.Array {
	q.Call(num.Copy(l.res, grad))
	return l.res
}

// utilities
func allocArray(a num.Array, dev num.Device, shape []int) num.Array {
	if a == nil || !num.SameShape(a.Dims(), shape) {
		a = num.NewArray(dev, num.Float32, shape...)
	}
	return a
}

func allocOnes(a num.Array, q num.Queue, size int) num.Array {
	if a == nil || a.Dims()[0] != size {
		a = num.NewArray(q.Device(), num.Float32, size)
		q.Call(num.Fill(a, 1))
	}
	return a
}

/*
func loadLayer(layer map[string]interface{}) Layer {
	switch layer["Type"].(string) {
	case "linear":
		return Linear(int(layer["Nout"].(float64)))
	case "activation":
		return Activation(layer["Atype"].(string))
	case "logRegression":
		return LogRegression()
	default:
		panic(fmt.Errorf("invalid layer type: %s", layer["Type"]))
	}
}
*/
