package nnet

import (
	"encoding/json"
	"fmt"
	"github.com/jnb666/deepthought2/num"
	"math/rand"
	"reflect"
	"strings"
)

// Layer interface type represents one layer of the neural net.
type Layer interface {
	Init(q num.Queue, inShape []int, opts num.LayerOpts, rng *rand.Rand) int
	InShape() []int
	OutShape() []int
	Fprop(q num.Queue, in *num.Array, work *num.Pool, trainMode bool) *num.Array
	Bprop(q num.Queue, grad, dsrc *num.Array, work *num.Pool) *num.Array
	Type() string
	ToString() string
	Output() *num.Array
	Memory() (weights, outputs, temp int)
	Release()
}

// ParamLayer is a layer with weight and bias parameters
type ParamLayer interface {
	Layer
	InitParams(q num.Queue, fn func() float64, bias float32)
	Params() (W, B *num.Array)
	ParamGrads() (dW, dB *num.Array)
	ParamVelocity() (vW, vB *num.Array)
	Copy(q num.Queue, layer Layer)
	FilterShape() []int
	BiasShape() []int
}

// BatchNormLayer stores the scale, shift, running mean and variance
type BatchNormLayer interface {
	ParamLayer
	Stats() (w, b, runMean, runVar *num.Array)
}

// OutputLayer is the final layer in the stack
type OutputLayer interface {
	Layer
	Loss(q num.Queue, yOneHot, yPred *num.Array) *num.Array
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
	case "pool":
		cfg := new(Pool)
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
	case "batchNorm":
		cfg := new(BatchNorm)
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
	Nfeats int
	Size   int
	Stride int
	Pad    bool
	NoBias bool
}

var convDefault = Conv{Stride: 1}

func (c Conv) Marshal() LayerConfig {
	if c.Stride == 0 {
		c.Stride = convDefault.Stride
	}
	return LayerConfig{Type: "conv", Data: marshal(c)}
}

func (c Conv) ToString() string {
	return "conv" + structToString(c, convDefault)
}

func (c Conv) Type() string { return "conv" }

func (c *Conv) unmarshal(data json.RawMessage) Layer {
	unmarshal(data, c)
	return &convDNN{Conv: *c}
}

// Max pooling layer, should follow conv layer.
type Pool struct {
	Size    int
	Stride  int
	Pad     bool
	Average bool
}

func (c Pool) Marshal() LayerConfig {
	if c.Stride == 0 {
		c.Stride = c.Size
	}
	return LayerConfig{Type: "pool", Data: marshal(c)}
}

func (c Pool) ToString() string {
	def := Pool{Stride: c.Size}
	if c.Average {
		return "pool" + structToString(c, def)
	} else {
		return "maxPool" + structToString(c, def)
	}
}

func (c Pool) Type() string { return "pool" }

func (c *Pool) unmarshal(data json.RawMessage) Layer {
	unmarshal(data, c)
	return &poolDNN{Pool: *c}
}

// Linear fully connected layer, implements ParamLayer interface.
type Linear struct{ Nout int }

func (c Linear) Marshal() LayerConfig {
	return LayerConfig{Type: "linear", Data: marshal(c)}
}

func (c Linear) ToString() string {
	return fmt.Sprintf("linear(%d)", c.Nout)
}

func (c Linear) Type() string { return "linear" }

func (c *Linear) unmarshal(data json.RawMessage) Layer {
	unmarshal(data, c)
	return &linear{Linear: *c}
}

// Sigmoid, tanh or relu activation layer, implements OutputLayer interface.
type Activation struct{ Atype string }

func (c Activation) Marshal() LayerConfig {
	return LayerConfig{Type: "activation", Data: marshal(c)}
}

func (c Activation) ToString() string { return c.Atype }

func (c Activation) Type() string { return "activation" }

func (c *Activation) unmarshal(data json.RawMessage) Layer {
	unmarshal(data, c)
	return &activation{Activation: *c}
}

// Dropout layer, randomly drops given ratio of nodes.
type Dropout struct{ Ratio float64 }

func (c Dropout) Marshal() LayerConfig {
	return LayerConfig{Type: "dropout", Data: marshal(c)}
}

func (c Dropout) ToString() string {
	return fmt.Sprintf("dropout(%g)", c.Ratio)
}

func (c Dropout) Type() string { return "dropout" }

func (c *Dropout) unmarshal(data json.RawMessage) Layer {
	unmarshal(data, c)
	return &dropout{Dropout: *c}
}

// Batch normalisation layer.
type BatchNorm struct{ AvgFactor, Epsilon float64 }

var batchNormDefault = BatchNorm{Epsilon: 0.001, AvgFactor: 0.1}

func (c BatchNorm) Marshal() LayerConfig {
	if c.Epsilon == 0 {
		c.Epsilon = batchNormDefault.Epsilon
	}
	if c.AvgFactor == 0 {
		c.AvgFactor = batchNormDefault.AvgFactor
	}
	return LayerConfig{Type: "batchNorm", Data: marshal(c)}
}

func (c BatchNorm) ToString() string {
	return "batchNorm" + structToString(c, batchNormDefault)
}

func (c BatchNorm) Type() string { return "batchNorm" }

func (c *BatchNorm) unmarshal(data json.RawMessage) Layer {
	unmarshal(data, c)
	return &batchNorm{BatchNorm: *c}
}

// Flatten layer reshapes from 4 to 2 dimensions.
type Flatten struct{}

func (c Flatten) Marshal() LayerConfig { return LayerConfig{Type: "flatten"} }

func (c Flatten) ToString() string { return "flatten" }

func (c Flatten) Type() string { return "flatten" }

// linear layer implementation
type linear struct {
	Linear
	paramBase
	opts     num.LayerOpts
	inShape  []int
	src, dst *num.Array
	ones     *num.Array
}

func (l *linear) InShape() []int { return l.inShape }

func (l *linear) OutShape() []int { return []int{l.Nout, l.inShape[1]} }

func (l *linear) FilterShape() []int { return []int{l.inShape[0], l.Nout} }

func (l *linear) BiasShape() []int { return []int{l.Nout} }

func (l *linear) Init(q num.Queue, inShape []int, opts num.LayerOpts, rng *rand.Rand) int {
	if len(inShape) != 2 {
		panic("Linear: expect 2 dimensional input")
	}
	l.opts = opts
	l.inShape = inShape
	l.paramBase = newParams(q, l.FilterShape(), l.BiasShape(), opts&num.MomentumUpdate != 0)
	l.dst = q.NewArray(num.Float32, l.Nout, inShape[1])
	wsize := inShape[1] * l.Nout
	if opts&num.BpropData != 0 {
		wsize = max(wsize, inShape[1]*inShape[0])
	}
	return wsize
}

func (l *linear) Memory() (weights, outputs, temp int) {
	return l.paramBase.memory(), num.Bytes(l.dst), num.Bytes(l.ones)
}

func (l *linear) Release() {
	l.paramBase.release()
	num.Release(l.dst, l.ones)
}

func (l *linear) Output() *num.Array { return l.dst }

func (l *linear) Fprop(q num.Queue, in *num.Array, work *num.Pool, trainMode bool) *num.Array {
	temp := work.NewArray(num.Float32, l.inShape[1], l.Nout)
	l.src = in
	q.Call(
		num.Copy(l.b, temp),
		num.Gemm(l.src, l.w, temp, num.Trans, num.NoTrans, true),
		num.Transpose(temp, l.dst),
	).Finish()
	work.Clear()
	return l.dst
}

func (l *linear) Bprop(q num.Queue, grad, dsrc *num.Array, work *num.Pool) *num.Array {
	if l.ones == nil {
		l.ones = q.NewArray(num.Float32, grad.Dims[1])
		q.Call(num.Fill(l.ones, 1))
	}
	q.Call(
		num.Gemv(grad, l.ones, l.db, num.NoTrans),
		num.Gemm(l.src, grad, l.dw, num.NoTrans, num.Trans, false),
	)
	if dsrc != nil {
		temp := work.NewArray(num.Float32, l.inShape[1], l.inShape[0])
		q.Call(
			num.Gemm(grad, l.w, temp, num.Trans, num.Trans, false),
			num.Transpose(temp, dsrc),
		).Finish()
		work.Clear()
	}
	return dsrc
}

// convolutional layer implementation
type convDNN struct {
	Conv
	paramBase
	num.ParamLayer
}

func (l *convDNN) Init(q num.Queue, inShape []int, opts num.LayerOpts, rng *rand.Rand) int {
	layer, workSize := num.NewConvLayer(q, opts, inShape, l.Nfeats, l.Size, l.Stride, l.Pad, l.NoBias)
	l.paramBase = newParams(q, layer.FilterShape(), layer.BiasShape(), opts&num.MomentumUpdate != 0)
	layer.SetParamData(l.w, l.b, l.dw, l.db)
	l.ParamLayer = layer
	return workSize
}

func (l *convDNN) Memory() (weights, outputs, temp int) {
	_, outputs, temp = l.ParamLayer.Memory()
	return l.paramBase.memory(), outputs, temp
}

func (l *convDNN) Release() {
	l.paramBase.release()
	l.ParamLayer.Release()
}

// pool layer implentation
type poolDNN struct {
	Pool
	num.Layer
}

func (l *poolDNN) Init(q num.Queue, inShape []int, opts num.LayerOpts, rng *rand.Rand) int {
	l.Layer = num.NewPoolLayer(q, opts, inShape, l.Size, l.Stride, l.Pad, l.Average)
	return 0
}

// activation layers
type activation struct {
	Activation
	num.Layer
	loss *num.Array
}

func (l *activation) Init(q num.Queue, inShape []int, opts num.LayerOpts, rng *rand.Rand) int {
	l.Layer = num.NewActivationLayer(q, l.Atype, inShape)
	return 0
}

func (l *activation) Memory() (weights, outputs, temp int) {
	_, outputs, temp = l.Layer.Memory()
	return 0, outputs, temp + num.Bytes(l.loss)
}

func (l *activation) Release() {
	l.Layer.Release()
	num.Release(l.loss)
}

func (l *activation) Loss(q num.Queue, yOneHot, yPred *num.Array) *num.Array {
	if l.loss == nil {
		l.loss = q.NewArray(num.Float32, l.InShape()...)
	}
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

func (l *dropout) Init(q num.Queue, inShape []int, opts num.LayerOpts, rng *rand.Rand) int {
	l.Layer = num.NewDropoutLayer(q, l.Ratio, inShape, rng.Int63())
	return 0
}

// batch normalisation layer implementation
type batchNorm struct {
	BatchNorm
	paramBase
	num.BatchNormLayer
}

func (l *batchNorm) Init(q num.Queue, inShape []int, opts num.LayerOpts, rng *rand.Rand) int {
	layer := num.NewBatchNormLayer(q, opts, l.AvgFactor, l.Epsilon, inShape)
	l.paramBase = newParams(q, layer.FilterShape(), layer.BiasShape(), opts&num.MomentumUpdate != 0)
	layer.SetParamData(l.w, l.b, l.dw, l.db)
	l.BatchNormLayer = layer
	return 0
}

func (l *batchNorm) InitParams(q num.Queue, fn func() float64, bias float32) {
	l.BatchNormLayer.InitParams(q)
}

func (l *batchNorm) Copy(q num.Queue, layer Layer) {
	if src, ok := layer.(*batchNorm); ok {
		wSrc, bSrc := src.Params()
		meanSrc, varSrc := src.Stats()
		wDst, bDst := l.Params()
		meanDst, varDst := l.Stats()
		q.Call(
			num.Copy(meanSrc, meanDst),
			num.Copy(varSrc, varDst),
			num.Copy(wSrc, wDst),
		)
		if bDst != nil {
			q.Call(num.Copy(bSrc, bDst))
		}
	} else {
		panic("invalid layer type for copy!")
	}
}

// flatten layer implementation
type flatten struct {
	Flatten
	inShape  []int
	outShape []int
	dst      *num.Array
}

func (l *flatten) InShape() []int { return l.inShape }

func (l *flatten) OutShape() []int { return l.outShape }

func (l *flatten) Init(q num.Queue, inShape []int, opts num.LayerOpts, rng *rand.Rand) int {
	l.inShape = inShape
	l.outShape = []int{num.Prod(l.inShape[:3]), l.inShape[3]}
	return 0
}

func (l *flatten) Memory() (weights, outputs, temp int) {
	return 0, 0, 0
}

func (l *flatten) Release() {}

func (l *flatten) Output() *num.Array { return l.dst }

func (l *flatten) Fprop(q num.Queue, in *num.Array, work *num.Pool, trainMode bool) *num.Array {
	l.dst = in.Reshape(l.outShape...)
	return l.dst
}

func (l *flatten) Bprop(q num.Queue, grad, dsrc *num.Array, work *num.Pool) *num.Array {
	if grad != nil && dsrc != nil {
		q.Call(num.Copy(grad.Reshape(l.inShape...), dsrc))
	}
	return dsrc
}

// weight and bias parameters
type paramBase struct {
	w, b   *num.Array
	dw, db *num.Array
	vw, vb *num.Array
}

func newParams(q num.Queue, filterShape, biasShape []int, momentum bool) paramBase {
	var p paramBase
	p.w = q.NewArray(num.Float32, filterShape...)
	p.dw = q.NewArray(num.Float32, filterShape...)
	if momentum {
		p.vw = q.NewArray(num.Float32, filterShape...)
	}
	if biasShape != nil {
		p.b = q.NewArray(num.Float32, biasShape...)
		p.db = q.NewArray(num.Float32, biasShape...)
		if momentum {
			p.vb = q.NewArray(num.Float32, biasShape...)
		}
	}
	return p
}

func (p paramBase) Params() (w, b *num.Array) {
	return p.w, p.b
}

func (p paramBase) ParamGrads() (dw, db *num.Array) {
	return p.dw, p.db
}

func (p paramBase) ParamVelocity() (vw, vb *num.Array) {
	return p.vw, p.vb
}

func (p paramBase) InitParams(q num.Queue, wInit func() float64, bias float32) {
	weights := make([]float32, num.Prod(p.w.Dims))
	for i := range weights {
		weights[i] = float32(wInit())
	}
	q.Call(num.Write(p.w, weights))
	if p.vw != nil {
		q.Call(num.Fill(p.vw, 0))
	}
	if p.b != nil {
		q.Call(num.Fill(p.b, bias))
		if p.vb != nil {
			q.Call(num.Fill(p.vb, 0))
		}
	}
}

func (p paramBase) Copy(q num.Queue, layer Layer) {
	if l, ok := layer.(ParamLayer); ok {
		W, B := l.Params()
		q.Call(num.Copy(W, p.w))
		if B != nil {
			q.Call(num.Copy(B, p.b))
		}
	} else {
		panic(fmt.Errorf("invalid layer type for copy: %T", layer))
	}
}

func (p paramBase) memory() int {
	return num.Bytes(p.w, p.b, p.dw, p.db, p.vw, p.vb)
}

func (p paramBase) release() {
	num.Release(p.w, p.b, p.dw, p.db, p.vw, p.vb)
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

func structToString(val, defVal interface{}) string {
	params := []string{}
	def := reflect.ValueOf(defVal)
	st := reflect.TypeOf(val)
	s := reflect.ValueOf(val)
	for i := 0; i < s.NumField(); i++ {
		val := s.Field(i).Interface()
		if !reflect.DeepEqual(val, def.Field(i).Interface()) {
			if _, isBool := val.(bool); isBool {
				params = append(params, st.Field(i).Name)
			} else if i == 0 {
				params = append(params, fmt.Sprint(val))
			} else {
				params = append(params, fmt.Sprintf("%s:%v", st.Field(i).Name, val))
			}
		}
	}
	if len(params) == 0 {
		return ""
	}
	return "(" + strings.Join(params, " ") + ")"
}
