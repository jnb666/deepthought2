// Package nnet contains routines for constructing, training and testing neural networks.
package nnet

import (
	"fmt"
	"github.com/jnb666/deepthought2/num"
	"math"
	"math/rand"
	"os"
	"strings"
	"time"
)

// Weight initialisation type
type InitType int

const (
	Zeros InitType = iota
	Ones
	RandomNormal
	RandomUniform
	LecunNormal
	GlorotUniform
)

func (t InitType) Options() []string {
	return []string{"Zeros", "Ones", "RandomNormal", "RandomUniform", "LecunNormal", "GlorotUniform"}
}

func (t InitType) String() string {
	return t.Options()[t]
}

func (t InitType) WeightFunc(dims []int, rng *rand.Rand) func() float64 {
	switch t {
	case Zeros:
		return func() float64 { return 0 }
	case Ones:
		return func() float64 { return 1 }
	case RandomNormal:
		return func() float64 { return rng.NormFloat64() }
	case RandomUniform:
		return func() float64 { return rng.Float64() }
	case LecunNormal:
		nin, _ := getSize(dims)
		scale := 1 / math.Sqrt(float64(nin))
		return func() float64 { return rng.NormFloat64() * scale }
	case GlorotUniform:
		nin, nout := getSize(dims)
		scale := math.Sqrt(6 / float64(nin+nout))
		return func() float64 { return (2*rng.Float64() - 1) * scale }
	default:
		panic("invalid InitType")
	}
}

func getSize(dims []int) (nin, nout int) {
	n := len(dims)
	size := 1
	if n == 4 {
		size = dims[0] * dims[1]
	}
	nin, nout = size*dims[n-2], size*dims[n-1]
	return
}

// Network type represents a multilayer neural network model.
type Network struct {
	Config
	Layers    []Layer
	WorkSpace num.Array
	queue     num.Queue
	classes   num.Array
	diffs     num.Array
	total     num.Array
	batchErr  num.Array
	batchLoss num.Array
	inputGrad num.Array
	inShape   []int
}

// New function creates a new network with the given layers.
func New(queue num.Queue, conf Config, batchSize int, inShape []int, rng *rand.Rand) *Network {
	n := &Network{Config: conf, queue: queue}
	n.allocArrays(batchSize)
	if conf.FlattenInput {
		n.inShape = []int{num.Prod(inShape), batchSize}
	} else {
		n.inShape = append(inShape, batchSize)
	}
	shape := n.inShape
	wsize := 0
	for ix, l := range conf.Layers {
		layer := l.Unmarshal()
		if size := layer.Init(queue, shape, ix, rng); size > wsize {
			wsize = size
		}
		n.Layers = append(n.Layers, layer)
		shape = layer.OutShape()
	}
	if wsize > 0 {
		n.WorkSpace = queue.NewArray(num.Float32, wsize)
	}
	return n
}

// release allocated buffers
func (n *Network) Release() {
	n.queue.Finish()
	for _, layer := range n.Layers {
		layer.Release()
	}
	n.classes.Release()
	n.diffs.Release()
	n.batchLoss.Release()
	n.batchErr.Release()
	n.total.Release()
	if n.WorkSpace != nil {
		n.WorkSpace.Release()
	}
	if n.inputGrad != nil {
		n.inputGrad.Release()
	}
}

// Initialise network weights using a linear or normal distribution.
// Weights for each layer are scaled by 1/sqrt(nin)
func (n *Network) InitWeights(rng *rand.Rand) {
	for i, layer := range n.Layers {
		if l, ok := layer.(ParamLayer); ok {
			W, _ := l.Params()
			dims := W.Dims()
			wInit := n.WeightInit
			if l.Type() == "batchNorm" {
				wInit = Ones
			}
			if n.DebugLevel >= 1 {
				fmt.Printf("layer %d: %s %v set weights init=%s bias=%.3g\n", i, l.Type(), dims, wInit, n.Bias)
			}
			l.InitParams(n.queue, wInit.WeightFunc(dims, rng), float32(n.Bias))
		}
	}
	if n.DebugLevel >= 2 {
		n.PrintWeights()
	}
}

// Copy weights and bias arrays to destination net
func (n *Network) CopyTo(net *Network, sync bool) {
	for i, layer := range n.Layers {
		if l, ok := layer.(ParamLayer); ok {
			W, B := l.Params()
			net.Layers[i].(ParamLayer).SetParams(n.queue, W, B)
		}
		if l, ok := layer.(BatchNormLayer); ok {
			runMean, runVar := l.Stats()
			net.Layers[i].(BatchNormLayer).SetStats(n.queue, runMean, runVar)
		}
	}
	if sync {
		n.queue.Finish()
	}
}

// Accessor for output layer
func (n *Network) OutLayer() OutputLayer {
	return n.Layers[len(n.Layers)-1].(OutputLayer)
}

// Feed forward the input to get the predicted output
func (n *Network) Fprop(input num.Array, trainMode bool) num.Array {
	pred := input
	for i, layer := range n.Layers {
		if n.DebugLevel >= 2 && pred != nil {
			fmt.Printf("layer %d input\n%s", i, pred.String(n.queue))
		}
		pred = layer.Fprop(n.queue, pred, n.WorkSpace, trainMode)
	}
	return pred
}

// Predict output given input data
func (n *Network) Predict(input, classes num.Array) num.Array {
	yPred := n.Fprop(input, false)
	if n.DebugLevel >= 2 {
		fmt.Printf("yPred\n%s", yPred.String(n.queue))
	}
	n.queue.Call(num.Unhot(yPred, classes))
	return yPred
}

// Calculate the error from the predicted versus actual values
// if pred slice is not nil then also return the predicted output classes.
func (n *Network) Error(dset *Dataset, pred []int32) float64 {
	n.queue.Call(num.Fill(n.total, 0))
	for batch := 0; batch < dset.Batches; batch++ {
		n.queue.Finish()
		x, y, _ := dset.NextBatch()
		n.Predict(x, n.classes)
		n.queue.Call(
			num.Neq(n.classes, y, n.diffs),
			num.Sum(n.diffs, n.batchErr),
			num.Axpy(1, n.batchErr, n.total),
		)
		if pred != nil {
			start := batch * dset.BatchSize
			end := start + y.Dims()[0]
			n.queue.Call(num.Read(n.classes, pred[start:end]))
		}
		if n.DebugLevel >= 2 || (n.DebugLevel >= 1 && batch == 0) {
			fmt.Printf("batch %d error =%s\n", batch, n.batchErr.String(n.queue))
			fmt.Println(y.String(n.queue))
			fmt.Println(n.classes.String(n.queue))
			fmt.Println(n.diffs.String(n.queue))
		}
	}
	err := []float32{0}
	n.queue.Call(num.Read(n.total, err)).Finish()
	return float64(err[0]) / float64(dset.Samples)
}

// Print network description
func (n *Network) String() string {
	s := n.configString()
	if n.Layers != nil {
		str := []string{"\n== Network ==    input " + fmt.Sprint(n.inShape[:len(n.inShape)-1])}
		for i, layer := range n.Layers {
			dims := layer.OutShape()
			str = append(str, fmt.Sprintf("%2d: %-12s %s", i, fmt.Sprint(dims[:len(dims)-1]), layer.ToString()))
		}
		s += strings.Join(str, "\n")
	}
	return s
}

// Print network weights
func (n *Network) PrintWeights() {
	for i, layer := range n.Layers {
		if l, ok := layer.(ParamLayer); ok {
			W, B := l.Params()
			fmt.Printf("== Layer %d weights ==\n%s %s\n", i, W.String(n.queue), B.String(n.queue))
		}
	}
}

func (n *Network) allocArrays(size int) {
	n.classes = n.queue.NewArray(num.Int32, size)
	n.diffs = n.queue.NewArray(num.Int32, size)
	n.batchLoss = n.queue.NewArray(num.Float32)
	n.batchErr = n.queue.NewArray(num.Float32)
	n.total = n.queue.NewArray(num.Float32)
}

// Set random number seed, or random seed if seed <= 0
func SetSeed(seed int64) *rand.Rand {
	if seed <= 0 {
		seed = time.Now().UTC().UnixNano()
	}
	fmt.Println("random seed =", seed)
	source := rand.NewSource(seed)
	return rand.New(source)
}

// Exit in case of error
func CheckErr(err error) {
	if err != nil {
		fmt.Fprintln(os.Stderr, err)
		os.Exit(1)
	}
}
