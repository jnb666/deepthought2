package nnet

import (
	"encoding/json"
	"fmt"
	"github.com/jnb666/deepthought2/num"
	"log"
	"math/rand"
	"reflect"
	"strings"
)

// Layer interface type represents one layer of the neural net.
type Layer interface {
	ConfigLayer
	Init(q num.Queue, inShape []int, opts num.LayerOpts, rng *rand.Rand) (workSize, inSize, weights int)
	InShape() []int
	OutShape() []int
	Fprop(q num.Queue, in *num.Array, work num.Buffer, trainMode bool) *num.Array
	Bprop(q num.Queue, grad, dsrc *num.Array, work [3]num.Buffer) *num.Array
	Output() *num.Array
	Memory() (weights, outputs, temp int)
	BpropData() bool
	Release()
}

// ParamLayer is a layer which may have weight and bias parameters
type ParamLayer interface {
	Layer
	Params() (W, B *num.Array)
	InitParams(q num.Queue, init InitType, bias float64, rng *rand.Rand)
	UpdateParams(q num.Queue, opt Optimiser, work num.Buffer)
	Copy(q num.Queue, layer Layer)
	Export(q num.Queue) []uint32
	Import(q num.Queue, vec []uint32)
	NumWeights() int
}

// LayerGroup is a compound layer made up of multiple layers
type LayerGroup interface {
	Layer
	Layers() []Layer
	LayerDesc() []string
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
	Type() string
	String() string
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
	case "add":
		cfg := new(Add)
		return cfg.unmarshal(l.Data)
	default:
		panic("invalid layer type: " + l.Type)
	}
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

func (c Conv) Type() string { return "conv" }

func (c Conv) String() string {
	if c.NoBias {
		c.NoBias = false
		return structToString("conv", c, convDefault)
	}
	return structToString("convBias", c, convDefault)
}

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

func (c Pool) Type() string { return "pool" }

func (c Pool) String() string {
	def := Pool{Stride: c.Size}
	return structToString("pool", c, def)
}

func (c *Pool) unmarshal(data json.RawMessage) Layer {
	unmarshal(data, c)
	return &poolDNN{Pool: *c}
}

// Linear fully connected layer, implements ParamLayer interface.
type Linear struct{ Nout int }

func (c Linear) Marshal() LayerConfig {
	return LayerConfig{Type: "linear", Data: marshal(c)}
}

func (c Linear) Type() string { return "linear" }

func (c Linear) String() string {
	var def Linear
	return structToString("linear", c, def)
}

func (c *Linear) unmarshal(data json.RawMessage) Layer {
	unmarshal(data, c)
	return &linear{Linear: *c}
}

// Sigmoid, tanh or relu activation layer, implements OutputLayer interface.
type Activation struct{ Atype string }

func (c Activation) Marshal() LayerConfig {
	return LayerConfig{Type: "activation", Data: marshal(c)}
}

func (c Activation) Type() string { return "activation" }

func (c Activation) String() string { return c.Atype }

func (c *Activation) unmarshal(data json.RawMessage) Layer {
	unmarshal(data, c)
	return &activation{Activation: *c}
}

// Dropout layer, randomly drops given ratio of nodes.
type Dropout struct{ Ratio float64 }

func (c Dropout) Marshal() LayerConfig {
	return LayerConfig{Type: "dropout", Data: marshal(c)}
}

func (c Dropout) Type() string { return "dropout" }

func (c Dropout) String() string {
	var def Dropout
	return structToString("dropout", c, def)
}

func (c *Dropout) unmarshal(data json.RawMessage) Layer {
	unmarshal(data, c)
	return &dropout{Dropout: *c}
}

// Batch normalisation layer.
type BatchNorm struct{ AvgFactor, Epsilon float64 }

var batchNormDefault = BatchNorm{Epsilon: 1e-4, AvgFactor: 0.1}

func (c BatchNorm) Marshal() LayerConfig {
	if c.Epsilon == 0 {
		c.Epsilon = batchNormDefault.Epsilon
	}
	if c.AvgFactor == 0 {
		c.AvgFactor = batchNormDefault.AvgFactor
	}
	return LayerConfig{Type: "batchNorm", Data: marshal(c)}
}

func (c BatchNorm) Type() string { return "batchNorm" }

func (c BatchNorm) String() string {
	return structToString("batchNorm", c, batchNormDefault)
}

func (c *BatchNorm) unmarshal(data json.RawMessage) Layer {
	unmarshal(data, c)
	return &batchNorm{BatchNorm: *c}
}

// Flatten layer reshapes from 4 to 2 dimensions.
type Flatten struct{}

func (c Flatten) Marshal() LayerConfig { return LayerConfig{Type: "flatten"} }

func (c Flatten) Type() string { return "flatten" }

func (c Flatten) String() string { return "flatten" }

// Add two sets of layers, used in residual network block, output is X(input) + Y(input)
// if Y is nil then outputput is X(input) + 1
type Add struct {
	X, Y []LayerConfig
}

func AddLayer(X, Y []ConfigLayer) Add {
	var c Add
	for _, l := range X {
		c.X = append(c.X, l.Marshal())
	}
	if Y != nil {
		for _, l := range Y {
			c.Y = append(c.Y, l.Marshal())
		}
	}
	return c
}

func (c Add) Marshal() LayerConfig {
	return LayerConfig{Type: "add", Data: marshal(c)}
}

func (c Add) Type() string { return "add" }

func (c Add) String() string { return "add" }

func (c *Add) unmarshal(data json.RawMessage) Layer {
	unmarshal(data, c)
	block := &add{Add: *c}
	for _, l := range c.X {
		block.layers[0] = append(block.layers[0], l.Unmarshal())
	}
	if c.Y == nil {
		block.layers[1] = []Layer{}
	} else {
		for _, l := range c.Y {
			block.layers[1] = append(block.layers[1], l.Unmarshal())
		}
	}
	return block
}

// linear layer implementation
type linear struct {
	Linear
	paramBase
	inShape   []int
	src, dst  *num.Array
	ones      *num.Array
	bpropData bool
}

func (l *linear) InShape() []int { return l.inShape }

func (l *linear) OutShape() []int { return []int{l.Nout, l.inShape[1]} }

func (l *linear) BpropData() bool { return l.bpropData }

func (l *linear) Init(q num.Queue, inShape []int, opts num.LayerOpts, rng *rand.Rand) (workSize, inSize, weights int) {
	if len(inShape) != 2 {
		panic("Linear: expect 2 dimensional input")
	}
	l.inShape = inShape
	l.paramBase = newParams(q, []int{inShape[0], l.Nout}, []int{l.Nout}, opts&num.MomentumUpdate != 0)
	l.dst = q.NewArray(num.Float32, l.Nout, inShape[1])
	workSize = inShape[1] * l.Nout
	if opts&num.BpropData != 0 {
		l.bpropData = true
		inSize = num.Prod(inShape)
		workSize = max(workSize, inShape[1]*inShape[0])
	}
	weights = l.NumWeights()
	return
}

func (l *linear) Memory() (weights, outputs, temp int) {
	return l.paramBase.memory(), num.Bytes(l.dst), num.Bytes(l.ones)
}

func (l *linear) Release() {
	l.paramBase.release()
	num.Release(l.dst, l.ones)
}

func (l *linear) Output() *num.Array { return l.dst }

func (l *linear) Fprop(q num.Queue, in *num.Array, work num.Buffer, trainMode bool) *num.Array {
	temp := num.NewArray(work, num.Float32, l.inShape[1], l.Nout)
	l.src = in
	q.Call(
		num.Copy(l.b, temp),
		num.Gemm(1, l.src, l.w, temp, num.Trans, num.NoTrans, true),
		num.Transpose(temp, l.dst),
	)
	return l.dst
}

func (l *linear) Bprop(q num.Queue, grad, dsrc *num.Array, work [3]num.Buffer) *num.Array {
	if l.ones == nil {
		l.ones = q.NewArray(num.Float32, grad.Dims[1])
		q.Call(num.Fill(l.ones, 1))
	}
	scale := 1.0 / float32(l.inShape[1])
	q.Call(
		num.Gemv(scale, grad, l.ones, l.db, num.NoTrans),
		num.Gemm(scale, l.src, grad, l.dw, num.NoTrans, num.Trans, false),
	)
	if l.bpropData {
		temp := num.NewArray(work[0], num.Float32, l.inShape[1], l.inShape[0])
		q.Call(
			num.Gemm(1, grad, l.w, temp, num.Trans, num.Trans, false),
			num.Transpose(temp, dsrc),
		)
	}
	return dsrc
}

// convolutional layer implementation
type convDNN struct {
	Conv
	paramBase
	num.ConvLayer
}

func (l *convDNN) Init(q num.Queue, inShape []int, opts num.LayerOpts, rng *rand.Rand) (workSize, inSize, weights int) {
	if l.NoBias {
		opts |= num.NoBias
	}
	l.ConvLayer, workSize = num.NewConvLayer(q, opts, inShape, l.Nfeats, l.Size, l.Stride, l.Pad)
	if debug >= 1 {
		log.Printf("    conv algorithm=%s\n", l.Algorithm())
	}
	l.paramBase = newParams(q, l.FilterShape(), l.BiasShape(), opts&num.MomentumUpdate != 0)
	l.SetParamData(l.w, l.b, l.dw, l.db)
	if opts&num.BpropData != 0 {
		inSize = num.Prod(inShape)
	}
	weights = l.NumWeights()
	return
}

func (l *convDNN) Memory() (weights, outputs, temp int) {
	_, outputs, temp = l.ConvLayer.Memory()
	return l.paramBase.memory(), outputs, temp
}

func (l *convDNN) Release() {
	l.paramBase.release()
	l.ConvLayer.Release()
}

// pool layer implentation
type poolDNN struct {
	Pool
	num.Layer
}

func (l *poolDNN) Init(q num.Queue, inShape []int, opts num.LayerOpts, rng *rand.Rand) (workSize, inSize, weights int) {
	l.Layer = num.NewPoolLayer(q, opts, inShape, l.Size, l.Stride, l.Pad, l.Average)
	if opts&num.BpropData != 0 {
		inSize = num.Prod(inShape)
	}
	return
}

// activation layers
type activation struct {
	Activation
	num.Layer
	loss *num.Array
}

func (l *activation) Init(q num.Queue, inShape []int, opts num.LayerOpts, rng *rand.Rand) (workSize, inSize, weights int) {
	l.Layer = num.NewActivationLayer(q, l.Atype, inShape)
	if opts&num.BpropData != 0 {
		inSize = num.Prod(inShape)
	}
	return
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

func (l *dropout) Init(q num.Queue, inShape []int, opts num.LayerOpts, rng *rand.Rand) (workSize, inSize, weights int) {
	seed := rng.Int63()
	if debug >= 1 {
		log.Printf("dropout: seed = %d\n", seed)
	}
	l.Layer = num.NewDropoutLayer(q, l.Ratio, inShape, seed)
	if opts&num.BpropData != 0 {
		inSize = num.Prod(inShape)
	}
	return
}

// batch normalisation layer implementation
type batchNorm struct {
	BatchNorm
	paramBase
	num.BatchNormLayer
}

func (l *batchNorm) Init(q num.Queue, inShape []int, opts num.LayerOpts, rng *rand.Rand) (workSize, inSize, weights int) {
	l.BatchNormLayer = num.NewBatchNormLayer(q, opts, l.AvgFactor, l.Epsilon, inShape)
	l.paramBase = newParams(q, l.FilterShape(), l.BiasShape(), opts&num.MomentumUpdate != 0)
	l.SetParamData(l.w, l.b, l.dw, l.db)
	if opts&num.BpropData != 0 {
		inSize = num.Prod(inShape)
	}
	weights = l.NumWeights()
	return
}

func (l *batchNorm) InitParams(q num.Queue, init InitType, bias float64, rng *rand.Rand) {
	l.BatchNormLayer.InitParams(q)
	if l.vw != nil {
		q.Call(num.Fill(l.vw, 0))
	}
	if l.vb != nil {
		q.Call(num.Fill(l.vb, 0))
	}
}

func (l *batchNorm) UpdateParams(q num.Queue, opt Optimiser, work num.Buffer) {
	if l.b != nil {
		//fmt.Printf("batchnorm w: %s\n", l.w.String(q))
		//fmt.Printf("batchnorm b: %s\n", l.b.String(q))
		opt.Update(q, true, l.w, l.dw, l.vw, work)
		opt.Update(q, false, l.b, l.db, l.vb, work)
	} else {
		//fmt.Printf("batchnorm w: %s\n", l.w.String(q))
		opt.Update(q, false, l.w, l.dw, l.vw, work)
	}
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

func (l *batchNorm) Export(q num.Queue) []uint32 {
	mean, variance := l.Stats()
	s1, s2, s3 := mean.Size(), variance.Size(), l.w.Size()
	vec := make([]uint32, l.NumWeights()+s1+s2)
	q.Call(
		num.Read(mean, vec[:s1]),
		num.Read(variance, vec[s1:s1+s2]),
		num.Read(l.w, vec[s1+s2:s1+s2+s3]),
	)
	if l.b != nil {
		q.Call(num.Read(l.b, vec[s1+s2+s3:]))
	}
	return vec
}

func (l *batchNorm) Import(q num.Queue, vec []uint32) {
	mean, variance := l.Stats()
	s1, s2, s3 := mean.Size(), variance.Size(), l.w.Size()
	if len(vec) != l.NumWeights()+s1+s2 {
		panic(fmt.Errorf("Import error: vec length=%d expecting %d", len(vec), l.NumWeights()+s1+s2))
	}
	q.Call(
		num.Write(mean, vec[:s1]),
		num.Write(variance, vec[s1:s1+s2]),
		num.Write(l.w, vec[s1+s2:s1+s2+s3]),
	)
	if l.b != nil {
		q.Call(num.Write(l.b, vec[s1+s2+s3:]))
	}
}

// flatten layer implementation
type flatten struct {
	Flatten
	inShape   []int
	outShape  []int
	dst       *num.Array
	bpropData bool
}

func (l *flatten) InShape() []int { return l.inShape }

func (l *flatten) OutShape() []int { return l.outShape }

func (l *flatten) BpropData() bool { return l.bpropData }

func (l *flatten) Init(q num.Queue, inShape []int, opts num.LayerOpts, rng *rand.Rand) (workSize, inSize, weights int) {
	l.inShape = inShape
	l.outShape = []int{num.Prod(l.inShape[:3]), l.inShape[3]}
	if opts&num.BpropData != 0 {
		l.bpropData = true
		inSize = num.Prod(inShape)
	}
	return
}

func (l *flatten) Memory() (weights, outputs, temp int) {
	return 0, 0, 0
}

func (l *flatten) Release() {}

func (l *flatten) Output() *num.Array { return l.dst }

func (l *flatten) Fprop(q num.Queue, in *num.Array, work num.Buffer, trainMode bool) *num.Array {
	l.dst = in.Reshape(l.outShape...)
	return l.dst
}

func (l *flatten) Bprop(q num.Queue, grad, dsrc *num.Array, work [3]num.Buffer) *num.Array {
	if grad != nil && dsrc != nil {
		q.Call(num.Copy(grad.Reshape(l.inShape...), dsrc))
	}
	return dsrc
}

// add layer implentation
type add struct {
	Add
	layers    [2][]Layer
	dst       *num.Array
	dsrc      *num.Array
	bpropData bool
}

func (l *add) Layers() []Layer {
	return append(l.layers[0], l.layers[1]...)
}

func (l *add) LayerDesc() []string {
	desc := make([]string, len(l.Layers()))
	for i := range desc {
		if i == 0 || i == len(l.layers[0]) {
			desc[i] = "->"
		} else {
			desc[i] = "  "
		}
	}
	return desc
}

func (l *add) InShape() []int { return l.layers[0][0].InShape() }

func (l *add) OutShape() []int { return l.layers[0][len(l.layers[0])-1].OutShape() }

func (l *add) BpropData() bool { return l.bpropData }

func (l *add) Init(q num.Queue, inShape []int, opts num.LayerOpts, rng *rand.Rand) (workSize, inSize, weights int) {
	var shape [2][]int
	for i, group := range l.layers {
		shape[i] = inShape
		for j, layer := range group {
			wSize, iSize, nWeight := layer.Init(q, shape[i], opts, rng)
			if debug >= 1 {
				log.Printf("  add layer %d:%d: %s %v => %v work=%d\n", i, j, layer.Type(), shape[i], layer.OutShape(), wSize)
			}
			workSize = max(workSize, wSize)
			inSize = max(inSize, iSize)
			weights = max(weights, nWeight)
			shape[i] = layer.OutShape()
		}
	}
	if !num.SameShape(shape[0], shape[1]) {
		panic(fmt.Errorf("add block output shapes should match: have %v %v", shape[0], shape[1]))
	}
	l.dst = q.NewArray(num.Float32, shape[0]...)
	if opts&num.BpropData != 0 {
		l.dsrc = q.NewArray(num.Float32, inShape...)
		l.bpropData = true
	}
	return
}

func (l *add) Memory() (weights, outputs, temp int) {
	return 0, num.Bytes(l.dst, l.dsrc), 0
}

func (l *add) Release() {
	for _, layer := range l.Layers() {
		layer.Release()
	}
	num.Release(l.dst, l.dsrc)
}

func (l *add) Output() *num.Array { return l.dst }

func (l *add) Fprop(q num.Queue, in *num.Array, work num.Buffer, trainMode bool) *num.Array {
	d1 := Fprop(q, l.layers[0], in, work, trainMode)
	d2 := in
	if len(l.layers[1]) > 0 {
		d2 = Fprop(q, l.layers[1], in, work, trainMode)
	}
	q.Call(
		num.Copy(d1, l.dst),
		num.Axpy(1, d2, l.dst),
	)
	return l.dst
}

func (l *add) Bprop(q num.Queue, grad, dsrc *num.Array, work [3]num.Buffer) *num.Array {
	g1 := Bprop(q, l.layers[0], grad, work)
	g2 := grad
	if len(l.layers[1]) > 0 {
		g2 = Bprop(q, l.layers[1], grad, work)
	}
	q.Call(
		num.Copy(g1, l.dsrc),
		num.Axpy(1, g2, l.dsrc),
	)
	return l.dsrc
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

func (p paramBase) InitParams(q num.Queue, init InitType, bias float64, rng *rand.Rand) {
	if debug >= 1 {
		log.Printf("set weights: dims=%v init=%s bias=%.3g\n", p.w.Dims, init, bias)
	}
	wInit := init.WeightFunc(p.w.Dims, rng)
	weights := make([]float32, num.Prod(p.w.Dims))
	for i := range weights {
		weights[i] = float32(wInit())
	}
	q.Call(num.Write(p.w, weights))
	if p.vw != nil {
		q.Call(num.Fill(p.vw, 0))
	}
	if p.b != nil {
		q.Call(num.Fill(p.b, float32(bias)))
		if p.vb != nil {
			q.Call(num.Fill(p.vb, 0))
		}
	}
}

func (p paramBase) UpdateParams(q num.Queue, opt Optimiser, work num.Buffer) {
	opt.Update(q, true, p.w, p.dw, p.vw, work)
	if p.b != nil {
		opt.Update(q, false, p.b, p.db, p.vb, work)
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

func (p paramBase) Export(q num.Queue) []uint32 {
	size := p.w.Size()
	vec := make([]uint32, p.NumWeights())
	q.Call(num.Read(p.w, vec[:size]))
	if p.b != nil {
		q.Call(num.Read(p.b, vec[size:]))
	}
	return vec
}

func (p paramBase) Import(q num.Queue, vec []uint32) {
	if len(vec) != p.NumWeights() {
		panic(fmt.Errorf("Import error: vec length=%d expecting %d", len(vec), p.NumWeights()))
	}
	size := p.w.Size()
	q.Call(num.Write(p.w, vec[:size]))
	if p.b != nil {
		q.Call(num.Write(p.b, vec[size:]))
	}
}

func (p paramBase) NumWeights() int {
	n := num.Prod(p.w.Dims)
	if p.b != nil {
		n += num.Prod(p.b.Dims)
	}
	return n
}

func (p paramBase) memory() int {
	return num.Bytes(p.w, p.b, p.dw, p.db, p.vw, p.vb)
}

func (p paramBase) release() {
	num.Release(p.w, p.b, p.dw, p.db, p.vw, p.vb)
}

func maxWeights(layers []Layer) int {
	nmax := 0
	for _, layer := range layers {
		if l, ok := layer.(ParamLayer); ok {
			if n := l.NumWeights(); n > nmax {
				nmax = n
			}
		}
	}
	return nmax
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

func structToString(name string, val, defVal interface{}) string {
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
		return name
	}
	return name + "(" + strings.Join(params, " ") + ")"
}
