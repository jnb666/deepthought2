package nnet

import (
	"encoding/json"
	"fmt"
	"github.com/jnb666/deepthought2/num"
	"math/rand"
)

// Layer interface type represents one layer of the neural net.
type Layer interface {
	Init(q num.Queue, inShape []int, index int, rng *rand.Rand) int
	InShape() []int
	OutShape() []int
	Fprop(q num.Queue, in, work num.Array) num.Array
	Bprop(q num.Queue, grad, work num.Array) num.Array
	IsActiv() bool
	Type() string
	ToString() string
	Output() num.Array
	Release()
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
	case "dropout":
		cfg := new(Dropout)
		return cfg.unmarshal(l.Data)
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

func (c Conv) Type() string {
	return "conv"
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

func (c MaxPool) Type() string {
	return "maxPool"
}

func (c MaxPool) IsActiv() bool {
	return false
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

func (c Linear) Type() string {
	return "linear"
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

func (c Activation) Type() string {
	return c.Atype
}

func (c Activation) IsActiv() bool {
	return true
}

func (c *Activation) unmarshal(data json.RawMessage) Layer {
	unmarshal(data, c)
	return &activation{Activation: *c}
}

// Dropout layer, randomly drops given ratio of nodes.
type Dropout struct {
	Ratio float64
}

func (c Dropout) Marshal() LayerConfig {
	return LayerConfig{Type: "dropout", Data: marshal(c)}
}

func (c Dropout) ToString() string {
	return fmt.Sprintf("dropout %+v", c)
}

func (c Dropout) Type() string {
	return "dropout"
}

func (c Dropout) IsActiv() bool {
	return false
}

func (c *Dropout) unmarshal(data json.RawMessage) Layer {
	unmarshal(data, c)
	return &dropout{Dropout: *c}
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
	inShape        []int
	src, dst, dsrc num.Array
	temp1, temp2   num.Array
	ones           num.Array
}

func (l *linear) InShape() []int { return l.inShape }

func (l *linear) OutShape() []int { return []int{l.Nout, l.inShape[1]} }

func (l *linear) Init(q num.Queue, inShape []int, ix int, rng *rand.Rand) int {
	if len(inShape) != 2 {
		panic("Linear: expect 2 dimensional input")
	}
	nBatch, nIn := inShape[1], inShape[0]
	l.inShape = inShape
	l.paramBase = newParams(q, []int{nIn, l.Nout}, nBatch)
	l.dst = q.NewArray(num.Float32, l.Nout, nBatch)
	l.temp1 = q.NewArray(num.Float32, nBatch, l.Nout)
	if ix > 0 {
		l.dsrc = q.NewArray(num.Float32, nIn, nBatch)
		l.temp2 = q.NewArray(num.Float32, nBatch, nIn)
	}
	return 0
}

func (l *linear) Release() {
	l.paramBase.release()
	l.dst.Release()
	l.temp1.Release()
	if l.dsrc != nil {
		l.dsrc.Release()
	}
	if l.temp2 != nil {
		l.temp2.Release()
	}
	if l.ones != nil {
		l.ones.Release()
	}
}

func (l *linear) Output() num.Array { return l.dst }

func (l *linear) Fprop(q num.Queue, in, work num.Array) num.Array {
	l.src = in
	q.Call(
		num.Copy(l.b, l.temp1),
		num.Gemm(l.src, l.w, l.temp1, num.Trans, num.NoTrans, true),
		num.Transpose(l.temp1, l.dst),
	)
	return l.dst
}

func (l *linear) Bprop(q num.Queue, grad, work num.Array) num.Array {
	if l.ones == nil {
		l.ones = q.NewArray(num.Float32, grad.Dims()[1])
		q.Call(num.Fill(l.ones, 1))
	}
	q.Call(
		num.Gemv(grad, l.ones, l.db, num.NoTrans),
		num.Gemm(l.src, grad, l.dw, num.NoTrans, num.Trans, false),
	)
	if l.dsrc != nil {
		q.Call(
			num.Gemm(grad, l.w, l.temp2, num.Trans, num.Trans, false),
			num.Transpose(l.temp2, l.dsrc),
		)
	}
	return l.dsrc
}

// convolutional layer implementation
type convDNN struct {
	Conv
	paramBase
	num.Layer
}

func (l *convDNN) Init(q num.Queue, inShape []int, ix int, rng *rand.Rand) int {
	layer, workSize := num.NewConvLayer(q, ix, inShape, l.Nfeats, l.Size, l.Stride, l.Pad)
	l.paramBase = newParams(q, layer.FilterShape(), inShape[3])
	layer.SetParamData(l.w, l.b, l.dw, l.db)
	l.Layer = layer
	return workSize
}

func (l *convDNN) Release() {
	l.paramBase.release()
	l.Layer.Release()
}

// pool layer implentation
type poolDNN struct {
	MaxPool
	num.Layer
}

func (l *poolDNN) Init(q num.Queue, inShape []int, ix int, rng *rand.Rand) int {
	l.Layer = num.NewMaxPoolLayer(q, inShape, l.Size, l.Stride)
	return 0
}

// activation layers
type activation struct {
	Activation
	num.Layer
	loss num.Array
}

func (l *activation) Init(q num.Queue, inShape []int, ix int, rng *rand.Rand) int {
	l.Layer = num.NewActivationLayer(q, l.Atype, inShape)
	l.loss = q.NewArray(num.Float32, inShape...)
	return 0
}

func (l *activation) Release() {
	l.Layer.Release()
	l.loss.Release()
}

func (l *activation) Loss(q num.Queue, yOneHot, yPred num.Array) num.Array {
	if l.Atype == "softmax" {
		q.Call(num.SoftmaxLoss(yOneHot, yPred, l.loss))
	} else {
		q.Call(num.QuadraticLoss(yOneHot, yPred, l.loss))
	}
	return l.loss
}

// dropout layer implementation
type dropout struct {
	Dropout
	num.Layer
}

func (l *dropout) IsActiv() bool { return false }

func (l *dropout) Init(q num.Queue, inShape []int, ix int, rng *rand.Rand) int {
	l.Layer = num.NewDropoutLayer(q, l.Ratio, inShape, rng.Int63())
	return 0
}

// flatten layer implementation
type flatten struct {
	inShape  []int
	outShape []int
	dst      num.Array
}

func (l *flatten) IsActiv() bool { return false }

func (l *flatten) Type() string { return "flatten" }

func (l *flatten) ToString() string { return "flatten" }

func (l *flatten) InShape() []int { return l.inShape }

func (l *flatten) OutShape() []int { return l.outShape }

func (l *flatten) Init(q num.Queue, inShape []int, ix int, rng *rand.Rand) int {
	l.inShape = inShape
	l.outShape = []int{num.Prod(l.inShape[:3]), l.inShape[3]}
	return 0
}

func (l *flatten) Release() {}

func (l *flatten) Output() num.Array { return l.dst }

func (l *flatten) Fprop(q num.Queue, in, work num.Array) num.Array {
	l.dst = in.Reshape(l.outShape...)
	return l.dst
}

func (l *flatten) Bprop(q num.Queue, grad, work num.Array) num.Array {
	return grad.Reshape(l.inShape...)
}

// weight and bias parameters
type paramBase struct {
	w, b   num.Array
	dw, db num.Array
	nBatch float32
}

func newParams(q num.Queue, filterShape []int, nBatch int) paramBase {
	nout := filterShape[len(filterShape)-1]
	return paramBase{
		w:      q.NewArray(num.Float32, filterShape...),
		b:      q.NewArray(num.Float32, nout),
		dw:     q.NewArray(num.Float32, filterShape...),
		db:     q.NewArray(num.Float32, nout),
		nBatch: float32(nBatch),
	}
}

func (p paramBase) release() {
	p.w.Release()
	p.b.Release()
	p.dw.Release()
	p.db.Release()
}

func (p paramBase) IsActiv() bool { return false }

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
