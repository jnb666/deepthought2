package nnet

import (
	"encoding/json"
	"fmt"
	"github.com/jnb666/deepthought2/num"
	"math/rand"
)

// Layer interface type represents one layer of the neural net.
type Layer interface {
	Init(q num.Queue, inShape []int, prev Layer) Layer
	Link(next Layer)
	OutShape(inShape []int) []int
	Fprop(in num.Array) num.Array
	Bprop(grad num.Array) num.Array
	ToString() string
}

// ParamLayer is a layer with weight and bias parameters
type ParamLayer interface {
	Layer
	InitParams(scale, bias float32, normal bool, rng *rand.Rand)
	Params() (W, B num.Array)
	ParamGrads() (dW, dB num.Array)
	SetParams(W, B num.Array)
	UpdateParams(learningRate, weightDecay float32)
}

// OutputLayer is the final layer in the stack
type OutputLayer interface {
	Layer
	Loss(yOneHot, yPred num.Array) num.Array
}

// LayerDNN hold a layer which implements the num.Layer interface
type LayerDNN interface {
	DNNLayer() num.Layer
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
	DNN  bool
}

func (c Linear) Marshal() LayerConfig {
	return LayerConfig{Type: "linear", Data: marshal(c)}
}

func (c Linear) ToString() string {
	return fmt.Sprintf("linear %+v", c)
}

func (c *Linear) unmarshal(data json.RawMessage) Layer {
	unmarshal(data, c)
	if c.DNN {
		return &linearDNN{Linear: *c}
	} else {
		return &linear{Linear: *c}
	}
}

// Sigmoid, tanh or relu activation layer, implements OutputLayer interface.
type Activation struct {
	Atype string
	DNN   bool
}

func (c Activation) Marshal() LayerConfig {
	return LayerConfig{Type: "activation", Data: marshal(c)}
}

func (c Activation) ToString() string {
	return fmt.Sprintf("activation %+v", c)
}

func (c *Activation) unmarshal(data json.RawMessage) Layer {
	unmarshal(data, c)
	if c.DNN {
		if c.Atype == "relu" {
			return &reluDNN{Activation: *c}
		} else {
			panic(fmt.Sprintf("DNN activation type %s invalid", c.Atype))
		}
	} else {
		layer := &activation{Activation: *c}
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
	layerBase
	paramBase
	temp1 num.Array
	temp2 num.Array
	ones  num.Array
}

func (l *linear) OutShape(inShape []int) []int {
	return []int{l.Nout, inShape[1]}
}

func (l *linear) Init(queue num.Queue, inShape []int, prev Layer) Layer {
	if len(inShape) != 2 {
		panic("Linear: expect 2 dimensional input")
	}
	nBatch, nIn := inShape[1], inShape[0]
	l.layerBase = newLayerBase(queue, inShape, l.OutShape(inShape))
	l.paramBase = newParams(queue, []int{nIn, l.Nout}, []int{l.Nout}, nBatch)
	l.ones = queue.NewArray(num.Float32, nBatch)
	l.temp1 = queue.NewArray(num.Float32, nBatch, nIn)
	l.temp2 = queue.NewArray(num.Float32, nBatch, l.Nout)
	queue.Call(num.Fill(l.ones, 1))
	return l
}

func (l *linear) Fprop(in num.Array) num.Array {
	l.src = in
	l.queue.Call(
		num.Copy(l.temp2, l.b),
		num.Gemm(1, 1, l.src, l.w, l.temp2, num.Trans, num.NoTrans),
		num.Transpose(l.temp2, l.dst),
	)
	return l.dst
}

func (l *linear) Bprop(grad num.Array) num.Array {
	l.queue.Call(
		num.Gemv(1, 0, grad, l.ones, l.db, num.NoTrans),
		num.Gemm(1, 0, l.src, grad, l.dw, num.NoTrans, num.Trans),
		num.Gemm(1, 0, grad, l.w, l.temp1, num.Trans, num.Trans),
		num.Transpose(l.temp1, l.dsrc),
	)
	return l.dsrc
}

type linearDNN struct {
	Linear
	paramBase
	*layerDNN
}

func (l *linearDNN) Init(queue num.Queue, inShape []int, prev Layer) Layer {
	if len(inShape) != 2 {
		panic("LinearDNN: expect 2 dimensional input")
	}
	nBatch, nIn := inShape[1], inShape[0]
	layer := queue.LinearLayer(nBatch, nIn, l.Nout)
	l.paramBase = newParams(queue, layer.FilterShape(), layer.BiasShape(), nBatch)
	layer.SetParams(l.w, l.b, l.dw, l.db)
	l.layerDNN = newLayerDNN(queue, layer)
	return l
}

// convolutional layer implementation
type convDNN struct {
	Conv
	paramBase
	*layerDNN
}

func (l *convDNN) Init(queue num.Queue, inShape []int, prev Layer) Layer {
	if len(inShape) != 4 {
		panic("ConvDNN: expect 4 dimensional input")
	}
	n, d, h, w := inShape[3], inShape[2], inShape[1], inShape[0]
	layer := queue.ConvLayer(n, d, h, w, l.Nfeats, l.Size, l.Stride, l.Pad)
	l.paramBase = newParams(queue, layer.FilterShape(), layer.BiasShape(), n)
	layer.SetParams(l.w, l.b, l.dw, l.db)
	l.layerDNN = newLayerDNN(queue, layer)
	return l
}

// pool layer implentation
type poolDNN struct {
	MaxPool
	*layerDNN
}

func (l *poolDNN) Init(queue num.Queue, inShape []int, prev Layer) Layer {
	if len(inShape) != 4 {
		panic("PoolDNN: expect 4 dimensional input")
	}
	layer := queue.MaxPoolLayer(prev.(LayerDNN).DNNLayer(), l.Size, l.Stride)
	l.layerDNN = newLayerDNN(queue, layer)
	return l
}

// activation layers
type activation struct {
	Activation
	layerBase
	activ func(x, y num.Array) num.Function
	deriv func(x, y, z num.Array) num.Function
	loss  num.Array
	queue num.Queue
}

func (l *activation) Init(queue num.Queue, inShape []int, prev Layer) Layer {
	l.queue = queue
	l.layerBase = newLayerBase(queue, inShape, inShape)
	l.loss = queue.NewArray(num.Float32, inShape...)
	return l
}

func (l *activation) Fprop(in num.Array) num.Array {
	l.src = in
	l.queue.Call(l.activ(l.src, l.dst))
	return l.dst
}

func (l *activation) Bprop(grad num.Array) num.Array {
	l.queue.Call(l.deriv(l.src, grad, l.dsrc))
	return l.dsrc
}

func (l *activation) Loss(yOneHot, yPred num.Array) num.Array {
	l.queue.Call(num.QuadraticLoss(yOneHot, yPred, l.loss))
	return l.loss
}

// reluDNN layer must always be preceded by another DNN layer such as linearDNN
type reluDNN struct {
	Activation
	*layerDNN
}

func (l *reluDNN) Init(queue num.Queue, inShape []int, prev Layer) Layer {
	layer := queue.ReluLayer(prev.(LayerDNN).DNNLayer())
	l.layerDNN = newLayerDNN(queue, layer)
	return l
}

// log regression output layer
type logRegression struct {
	layerBase
	loss  num.Array
	queue num.Queue
}

func (l *logRegression) ToString() string { return fmt.Sprintf("logRegression") }

func (l *logRegression) Init(queue num.Queue, inShape []int, prev Layer) Layer {
	l.queue = queue
	l.layerBase = newLayerBase(queue, inShape, inShape)
	l.loss = queue.NewArray(num.Float32, inShape...)
	return l
}

func (l *logRegression) Fprop(in num.Array) num.Array {
	l.src = in
	l.queue.Call(num.Softmax(l.src, l.dst))
	return l.dst
}

func (l *logRegression) Bprop(grad num.Array) num.Array {
	l.queue.Call(num.Copy(l.dsrc, grad))
	return l.dsrc
}

func (l *logRegression) Loss(yOneHot, yPred num.Array) num.Array {
	l.queue.Call(num.SoftmaxLoss(yOneHot, yPred, l.loss))
	return l.loss
}

type flatten struct {
	layerBase
}

func (l *flatten) ToString() string { return fmt.Sprintf("flatten") }

func (l *flatten) OutShape(inShape []int) []int {
	return []int{num.Prod(inShape[:3]), inShape[3]}
}

func (l *flatten) Init(queue num.Queue, inShape []int, prev Layer) Layer {
	return l
}

func (l *flatten) Fprop(in num.Array) num.Array {
	l.src = in
	dims := in.Dims()
	l.dst = in.Reshape(-1, dims[len(dims)-1])
	return l.dst
}

func (l *flatten) Bprop(grad num.Array) num.Array {
	l.dsrc = grad.Reshape(l.src.Dims()...)
	return l.dsrc
}

// base blas layer type
type layerBase struct {
	src  num.Array
	dst  num.Array
	dsrc num.Array
}

func newLayerBase(queue num.Queue, inShape, outShape []int) layerBase {
	return layerBase{
		dst:  queue.NewArray(num.Float32, outShape...),
		dsrc: queue.NewArray(num.Float32, inShape...),
	}
}

func (l layerBase) OutShape(inShape []int) []int { return inShape }

func (l layerBase) Link(next Layer) {}

type layerDNN struct {
	que   num.Queue
	layer num.Layer
	layerBase
}

func newLayerDNN(queue num.Queue, layer num.Layer) *layerDNN {
	l := &layerDNN{que: queue, layer: layer}
	l.dsrc = l.layer.DiffSrc()
	return l
}

func (l *layerDNN) Link(next Layer) {
	l.dst = l.layer.Dst()
}

func (l *layerDNN) DNNLayer() num.Layer {
	return l.layer
}

func (l *layerDNN) OutShape(inShape []int) []int {
	return l.layer.OutShape()
}

func (l *layerDNN) Fprop(in num.Array) num.Array {
	l.layer.SetSrc(in)
	l.que.Call(num.Fprop(l.layer))
	return l.dst
}

func (l *layerDNN) Bprop(grad num.Array) num.Array {
	l.layer.SetDiffDst(grad)
	l.que.Call(num.BpropData(l.layer))
	if l.layer.HasParams() {
		l.que.Call(
			num.BpropFilter(l.layer),
			num.BpropBias(l.layer),
		)
	}
	return l.dsrc
}

// weight and bias parameters
type paramBase struct {
	queue  num.Queue
	w, b   num.Array
	dw, db num.Array
	nBatch float32
}

func newParams(queue num.Queue, wShape, bShape []int, nBatch int) paramBase {
	return paramBase{
		queue:  queue,
		w:      queue.NewArray(num.Float32, wShape...),
		b:      queue.NewArray(num.Float32, bShape...),
		dw:     queue.NewArray(num.Float32, wShape...),
		db:     queue.NewArray(num.Float32, bShape...),
		nBatch: float32(nBatch),
	}
}

func (p paramBase) Params() (W, B num.Array) {
	return p.w, p.b
}

func (p paramBase) ParamGrads() (dW, dB num.Array) {
	return p.dw, p.db
}

func (p paramBase) InitParams(scale, bias float32, normal bool, rng *rand.Rand) {
	weights := make([]float32, num.Prod(p.w.Dims()))
	for i := range weights {
		if normal {
			weights[i] = float32(rng.NormFloat64()) * scale
		} else {
			weights[i] = rng.Float32() * scale
		}
	}
	p.queue.Call(
		num.Write(p.w, weights),
		num.Fill(p.b, bias),
	)
}

func (p paramBase) SetParams(W, B num.Array) {
	p.queue.Call(num.Copy(p.w, W), num.Copy(p.b, B))
}

func (p paramBase) UpdateParams(learningRate, weightDecay float32) {
	if weightDecay != 0 {
		p.queue.Call(num.Axpy(-weightDecay*p.nBatch, p.w, p.dw))
	}
	p.queue.Call(
		num.Axpy(-learningRate/p.nBatch, p.dw, p.w),
		num.Axpy(-learningRate/p.nBatch, p.db, p.b),
	)
}

func isDNNLayer(l Layer) bool {
	if l == nil {
		return false
	}
	_, ok := l.(LayerDNN)
	return ok
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
