package nnet

import (
	"encoding/json"
	"fmt"
	"github.com/jnb666/deepthought2/num"
	"math/rand"
)

// Layer interface type represents one layer of the neural net.
type Layer interface {
	Init(dev num.Device, inShape []int, prev Layer) Layer
	InShape() []int
	OutShape() []int
	Fprop(q num.Queue, in num.Array) num.Array
	Bprop(q num.Queue, grad num.Array) num.Array
	ToString() string
	DNNLayer() num.Layer
}

// ParamLayer is a layer with weight and bias parameters
type ParamLayer interface {
	Layer
	InitParams(q num.Queue, scale, bias float32, normal bool, rng *rand.Rand)
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

// Layer configuration details
type LayerConfig struct {
	Type string
	Data json.RawMessage
}

type ConfigLayer interface {
	Marshal() LayerConfig
}

// Unmarshal JSON data and construct new layer
func (l LayerConfig) Unmarshal() Layer {
	switch l.Type {
	case "conv":
		cfg := new(Conv)
		return cfg.unmarshal(l.Data)
	case "maxPool":
		cfg := new(MaxPool)
		return cfg.unmarshal(l.Data)
	case "linear":
		cfg := new(Linear)
		return cfg.unmarshal(l.Data)
	case "activation":
		cfg := new(Activation)
		return cfg.unmarshal(l.Data)
	case "logRegression":
		return &logRegression{}
	case "flatten":
		return &flatten{}
	default:
		panic("invalid layer type: " + l.Type)
	}
}

func (l LayerConfig) String() string {
	return l.Unmarshal().ToString()
}

// Convolutional layer, implements ParamLayer interface.
type Conv struct {
	Nfeats, Size, Stride, Pad int
}

func (c Conv) Marshal() LayerConfig {
	if c.Stride == 0 {
		c.Stride = 1
	}
	return LayerConfig{Type: "conv", Data: marshal(c)}
}

func (c Conv) ToString() string {
	return fmt.Sprintf("conv %+v", c)
}

func (c *Conv) unmarshal(data json.RawMessage) Layer {
	unmarshal(data, c)
	return &convDNN{Conv: *c}
}

// Max pooling layer, should follow conv layer.
type MaxPool struct {
	Size, Stride int
}

func (c MaxPool) Marshal() LayerConfig {
	if c.Stride == 0 {
		c.Stride = c.Size
	}
	return LayerConfig{Type: "maxPool", Data: marshal(c)}
}

func (c MaxPool) ToString() string {
	return fmt.Sprintf("maxPool %+v", c)
}

func (c *MaxPool) unmarshal(data json.RawMessage) Layer {
	unmarshal(data, c)
	return &poolDNN{MaxPool: *c}
}

// Linear fully connected layer, implements ParamLayer interface.
type Linear struct {
	Nout int
}

func (c Linear) Marshal() LayerConfig {
	return LayerConfig{Type: "linear", Data: marshal(c)}
}

func (c Linear) ToString() string {
	return fmt.Sprintf("linear %+v", c)
}

func (c *Linear) unmarshal(data json.RawMessage) Layer {
	unmarshal(data, c)
	return &linear{Linear: *c}
}

// Sigmoid, tanh or relu activation layer, implements OutputLayer interface.
type Activation struct {
	Atype string
}

func (c Activation) Marshal() LayerConfig {
	return LayerConfig{Type: "activation", Data: marshal(c)}
}

func (c Activation) ToString() string {
	return fmt.Sprintf("activation %+v", c)
}

func (c *Activation) unmarshal(data json.RawMessage) Layer {
	unmarshal(data, c)
	return &activation{Activation: *c}
}

// LogRegression output layer with soft max activation.
type LogRegression struct{}

func (c LogRegression) Marshal() LayerConfig {
	return LayerConfig{Type: "logRegression"}
}

// Flatten layer reshapes from 4 to 2 dimensions.
type Flatten struct{}

func (c Flatten) Marshal() LayerConfig {
	return LayerConfig{Type: "flatten"}
}

// linear layer implementation
type linear struct {
	Linear
	paramBase
	src, dst, dsrc num.Array
	temp1, temp2   num.Array
	ones           num.Array
}

func (l *linear) InShape() []int { return l.dsrc.Dims() }

func (l *linear) OutShape() []int { return l.dst.Dims() }

func (l *linear) DNNLayer() num.Layer { return nil }

func (l *linear) Init(dev num.Device, inShape []int, prev Layer) Layer {
	if len(inShape) != 2 {
		panic("Linear: expect 2 dimensional input")
	}
	nBatch, nIn := inShape[1], inShape[0]
	l.paramBase = newParams(dev, []int{nIn, l.Nout}, []int{l.Nout}, nBatch)
	l.dst = dev.NewArray(num.Float32, l.Nout, nBatch)
	l.dsrc = dev.NewArray(num.Float32, nIn, nBatch)
	l.temp1 = dev.NewArray(num.Float32, nBatch, nIn)
	l.temp2 = dev.NewArray(num.Float32, nBatch, l.Nout)
	return l
}

func (l *linear) Fprop(q num.Queue, in num.Array) num.Array {
	l.src = in
	q.Call(
		num.Copy(l.b, l.temp2),
		num.Gemm(1, 1, l.src, l.w, l.temp2, num.Trans, num.NoTrans),
		num.Transpose(l.temp2, l.dst),
	)
	return l.dst
}

func (l *linear) Bprop(q num.Queue, grad num.Array) num.Array {
	if l.ones == nil {
		l.ones = q.NewArray(num.Float32, grad.Dims()[1])
		q.Call(num.Fill(l.ones, 1))
	}
	q.Call(
		num.Gemv(1, 0, grad, l.ones, l.db, num.NoTrans),
		num.Gemm(1, 0, l.src, grad, l.dw, num.NoTrans, num.Trans),
		num.Gemm(1, 0, grad, l.w, l.temp1, num.Trans, num.Trans),
		num.Transpose(l.temp1, l.dsrc),
	)
	return l.dsrc
}

// convolutional layer implementation
type convDNN struct {
	Conv
	paramBase
	num.Layer
}

func (l *convDNN) Init(dev num.Device, inShape []int, prev Layer) Layer {
	if len(inShape) != 4 {
		panic("ConvDNN: expect 4 dimensional input")
	}
	n, d, h, w := inShape[3], inShape[2], inShape[1], inShape[0]
	l.Layer = num.ConvLayer(dev, n, d, h, w, l.Nfeats, l.Size, l.Stride, l.Pad)
	l.paramBase = newParams(dev, l.FilterShape(), l.BiasShape(), n)
	l.SetParamData(l.w, l.b, l.dw, l.db)
	return l
}

// pool layer implentation
type poolDNN struct {
	MaxPool
	num.Layer
}

func (l *poolDNN) Init(dev num.Device, inShape []int, prev Layer) Layer {
	if len(inShape) != 4 {
		panic("PoolDNN: expect 4 dimensional input")
	}
	l.Layer = num.MaxPoolLayer(dev, prev.DNNLayer(), l.Size, l.Stride)
	return l
}

// activation layers
type activation struct {
	Activation
	num.Layer
	loss num.Array
}

func (l *activation) Init(dev num.Device, inShape []int, prev Layer) Layer {
	l.Layer = num.ActivationLayer(dev, l.Atype, inShape, prev.DNNLayer())
	l.loss = dev.NewArray(num.Float32, inShape...)
	return l
}

func (l *activation) Loss(q num.Queue, yOneHot, yPred num.Array) num.Array {
	q.Call(num.QuadraticLoss(yOneHot, yPred, l.loss))
	return l.loss
}

// log regression output layer
type logRegression struct {
	dst  num.Array
	dsrc num.Array
	loss num.Array
}

func (l *logRegression) ToString() string { return fmt.Sprintf("logRegression") }

func (l *logRegression) Init(dev num.Device, inShape []int, prev Layer) Layer {
	l.dst = dev.NewArray(num.Float32, inShape...)
	l.dsrc = dev.NewArray(num.Float32, inShape...)
	l.loss = dev.NewArray(num.Float32, inShape...)
	return l
}

func (l *logRegression) InShape() []int { return l.dst.Dims() }

func (l *logRegression) OutShape() []int { return l.dst.Dims() }

func (l *logRegression) DNNLayer() num.Layer { return nil }

func (l *logRegression) Fprop(q num.Queue, in num.Array) num.Array {
	q.Call(num.Softmax(in, l.dst))
	return l.dst
}

func (l *logRegression) Bprop(q num.Queue, grad num.Array) num.Array {
	q.Call(num.Copy(grad, l.dsrc))
	return l.dsrc
}

func (l *logRegression) Loss(q num.Queue, yOneHot, yPred num.Array) num.Array {
	q.Call(num.SoftmaxLoss(yOneHot, yPred, l.loss))
	return l.loss
}

type flatten struct {
	inShape  []int
	outShape []int
}

func (l *flatten) ToString() string { return fmt.Sprintf("flatten") }

func (l *flatten) InShape() []int { return l.inShape }

func (l *flatten) OutShape() []int { return l.outShape }

func (l *flatten) DNNLayer() num.Layer { return nil }

func (l *flatten) Init(dev num.Device, inShape []int, prev Layer) Layer {
	l.inShape = inShape
	l.outShape = []int{num.Prod(l.inShape[:3]), l.inShape[3]}
	return l
}

func (l *flatten) Fprop(q num.Queue, in num.Array) num.Array {
	return in.Reshape(l.outShape...)
}

func (l *flatten) Bprop(q num.Queue, grad num.Array) num.Array {
	return grad.Reshape(l.inShape...)
}

// weight and bias parameters
type paramBase struct {
	w, b   num.Array
	dw, db num.Array
	nBatch float32
}

func newParams(dev num.Device, wShape, bShape []int, nBatch int) paramBase {
	return paramBase{
		w:      dev.NewArray(num.Float32, wShape...),
		b:      dev.NewArray(num.Float32, bShape...),
		dw:     dev.NewArray(num.Float32, wShape...),
		db:     dev.NewArray(num.Float32, bShape...),
		nBatch: float32(nBatch),
	}
}

func (p paramBase) Params() (W, B num.Array) {
	return p.w, p.b
}

func (p paramBase) ParamGrads() (dW, dB num.Array) {
	return p.dw, p.db
}

func (p paramBase) InitParams(q num.Queue, scale, bias float32, normal bool, rng *rand.Rand) {
	weights := make([]float32, num.Prod(p.w.Dims()))
	for i := range weights {
		if normal {
			weights[i] = float32(rng.NormFloat64()) * scale
		} else {
			weights[i] = rng.Float32() * scale
		}
	}
	q.Call(
		num.Write(p.w, weights),
		num.Fill(p.b, bias),
	)
}

func (p paramBase) SetParams(q num.Queue, W, B num.Array) {
	q.Call(num.Copy(W, p.w), num.Copy(B, p.b))
}

func (p paramBase) UpdateParams(q num.Queue, learningRate, weightDecay float32) {
	if weightDecay != 0 {
		q.Call(num.Axpy(-weightDecay*p.nBatch, p.w, p.dw))
	}
	q.Call(
		num.Axpy(-learningRate/p.nBatch, p.dw, p.w),
		num.Axpy(-learningRate/p.nBatch, p.db, p.b),
	)
}

func marshal(v interface{}) []byte {
	data, err := json.Marshal(v)
	if err != nil {
		panic(err)
	}
	return data
}

func unmarshal(data json.RawMessage, v interface{}) {
	err := json.Unmarshal(data, v)
	if err != nil {
		panic(err)
	}
}
