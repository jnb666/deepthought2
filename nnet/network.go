// Package nnet contains routines for constructing, training and testing neural networks.
package nnet

import (
	"fmt"
	"github.com/jnb666/deepthought2/num"
	"log"
	"math"
	"math/rand"
	"os"
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
	HeNormal
)

func (t InitType) Options() []string {
	return []string{"Zeros", "Ones", "RandomNormal", "RandomUniform", "LecunNormal", "GlorotUniform", "HeNormal"}
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
	case GlorotUniform:
		nin, nout := getSize(dims)
		scale := math.Sqrt(6 / float64(nin+nout))
		return func() float64 { return (2*rng.Float64() - 1) * scale }
	case LecunNormal:
		nin, _ := getSize(dims)
		scale := 1 / math.Sqrt(float64(nin))
		return truncatedNormal(scale, 1, rng)
	case HeNormal:
		nin, _ := getSize(dims)
		scale := 2 / math.Sqrt(float64(nin))
		return truncatedNormal(scale, 1, rng)
	default:
		panic("invalid InitType")
	}
}

func truncatedNormal(scale, clip float64, rng *rand.Rand) func() float64 {
	return func() float64 {
		x := 2 * clip
		for math.Abs(x) > clip {
			x = rng.NormFloat64() * scale
		}
		return x
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
	batchErr  num.Array
	totalErr  num.Array
	batchLoss num.Array
	totalLoss num.Array
	inputGrad num.Array
	inShape   []int
	workSize  []int
}

// New function creates a new network with the given layers.
func New(queue num.Queue, conf Config, batchSize int, inShape []int, rng *rand.Rand) *Network {
	n := &Network{Config: conf, queue: queue}
	n.allocArrays(batchSize)
	n.inShape = append(inShape, batchSize)
	shape := n.inShape
	layerId := 0
	n.workSize = make([]int, len(conf.Layers))
	for i, l := range conf.Layers {
		layer := l.Unmarshal()
		n.workSize[i] = layer.Init(queue, shape, layerId, conf.Momentum != 0, rng)
		n.Layers = append(n.Layers, layer)
		shape = layer.OutShape()
		if l.Type != "flatten" {
			layerId++
		}
	}
	if wsize := max(n.workSize); wsize > 0 {
		n.WorkSpace = queue.NewArray(num.Float32, wsize/4)
	}
	return n
}

// release allocated buffers
func (n *Network) Release() {
	n.queue.Finish()
	for _, layer := range n.Layers {
		layer.Release()
	}
	num.Release(n.classes, n.diffs, n.batchLoss, n.batchErr, n.totalLoss, n.totalErr, n.WorkSpace, n.inputGrad)
}

// Initialise network weights using a linear or normal distribution.
// Weights for each layer are scaled by 1/sqrt(nin)
func (n *Network) InitWeights(rng *rand.Rand) {
	for i, layer := range n.Layers {
		if l, ok := layer.(ParamLayer); ok {
			dims := l.FilterShape()
			if n.DebugLevel >= 1 {
				log.Printf("layer %d: %s %v set weights init=%s bias=%.3g\n", i, l.Type(), dims, n.WeightInit, n.Bias)
			}
			l.InitParams(n.queue, n.WeightInit.WeightFunc(dims, rng), float32(n.Bias))
		}
	}
	if n.DebugLevel >= 2 {
		n.PrintWeights()
	}
}

// Copy weights and bias arrays to destination net
func (n *Network) CopyTo(net *Network, sync bool) {
	for i, layer := range n.Layers {
		if l, ok := net.Layers[i].(ParamLayer); ok {
			l.Copy(n.queue, layer)
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
			log.Printf("layer %d input\n%s", i, pred.String(n.queue))
		}
		pred = layer.Fprop(n.queue, pred, n.WorkSpace, trainMode)
	}
	return pred
}

// Get difference at output, back propagate gradient and update weights
func (n *Network) Bprop(batch int, yPred, yOneHot num.Array, opt Optimiser) {
	q := n.queue
	q.Call(
		num.Copy(yPred, n.inputGrad),
		num.Axpy(-1, yOneHot, n.inputGrad),
	)
	if n.DebugLevel >= 2 || (n.DebugLevel == 1 && batch == 0) {
		log.Printf("input grad:\n%s", n.inputGrad.String(q))
	}
	grad := n.inputGrad
	for i := len(n.Layers) - 1; i >= 0; i-- {
		grad = n.Layers[i].Bprop(q, grad, n.WorkSpace)
		if n.DebugLevel >= 3 && grad != nil {
			log.Printf("layer %d bprop output:\n%s", i, grad.String(q))
		}
	}
	for _, layer := range n.Layers {
		if l, ok := layer.(ParamLayer); ok {
			W, B, dW, dB := l.Params()
			vW, vB, vWPrev, vBPrev := l.ParamVelocity()
			opt.Update(q, true, W, dW, vW, vWPrev)
			if B != nil {
				opt.Update(q, false, B, dB, vB, vBPrev)
			}
		}
	}
}

// get the loss for this batch
func (n *Network) calcBatchLoss(batch int, yOneHot, yPred num.Array) {
	q := n.queue
	losses := n.OutLayer().Loss(q, yOneHot, yPred)
	if n.DebugLevel >= 2 {
		log.Printf("loss:\n%s", losses.String(q))
	}
	q.Call(
		num.Sum(losses, n.batchLoss),
		num.Axpy(1, n.batchLoss, n.totalLoss),
	)
}

// get the error for this batch, if pred is non-null then save predicted values
func (n *Network) calcBatchError(batch int, dset *Dataset, y, yPred num.Array, pred []int32) {
	q := n.queue
	q.Call(
		num.Unhot(yPred, n.classes),
		num.Neq(n.classes, y, n.diffs),
		num.Sum(n.diffs, n.batchErr),
		num.Axpy(1, n.batchErr, n.totalErr),
	)
	if n.DebugLevel >= 1 || (n.DebugLevel >= 1 && batch == 0) {
		log.Printf("batch %d error =%s\n", batch, n.batchErr.String(q))
		log.Println(y.String(q))
		log.Println(n.classes.String(q))
		log.Println(n.diffs.String(q))
	}
	if pred != nil {
		start := batch * dset.BatchSize
		end := start + dset.BatchSize
		if end > dset.Samples {
			end = dset.Samples
		}
		q.Call(num.Read(n.classes, pred[start:end]))
	}
}

// Calculate the error from the predicted versus actual values
// if pred slice is not nil then also return the predicted output classes.
func (n *Network) Error(dset *Dataset, pred []int32) float64 {
	dset.NextEpoch()
	n.queue.Call(num.Fill(n.totalErr, 0))
	var p []int32
	if pred != nil {
		p = make([]int32, dset.Samples)
	}
	for batch := 0; batch < dset.Batches; batch++ {
		n.queue.Finish()
		x, y, _ := dset.NextBatch()
		yPred := n.Fprop(x, false)
		if n.DebugLevel >= 2 {
			log.Printf("yPred\n%s", yPred.String(n.queue))
		}
		n.calcBatchError(batch, dset, y, yPred, p)
	}
	err := []float32{0}
	n.queue.Call(num.Read(n.totalErr, err)).Finish()
	if pred != nil {
		for i, ix := range dset.indexes {
			pred[ix] = p[i]
		}
	}
	return float64(err[0]) / float64(dset.Samples)
}

// Print network description
func (n *Network) String() string {
	s := n.configString()
	if n.Layers != nil {
		s += fmt.Sprintf("\n== Network ==\n    %-12s input", fmt.Sprint(n.inShape[:len(n.inShape)-1]))
		for i, layer := range n.Layers {
			dims := layer.OutShape()
			weights := ""
			if l, ok := layer.(ParamLayer); ok {
				n := num.Prod(l.FilterShape())
				if l.BiasShape() != nil {
					n += num.Prod(l.BiasShape())
				}
				weights = fmt.Sprint(n)
			}
			s += fmt.Sprintf("\n%2d: %-12s %s %s", i, fmt.Sprint(dims[:len(dims)-1]), layer.ToString(), weights)
		}
	}
	return s
}

// Get total allocated memory in bytes
func (n *Network) Memory() int {
	_, total := n.meminfo()
	return total[3]
}

// Print profile of allocated memory
func (n *Network) MemoryProfile() string {
	mem, total := n.meminfo()
	s := fmt.Sprintf("== memory profile ==                weights  outputs     temp    total (%d)\n",
		n.inShape[len(n.inShape)-1])
	for i, layer := range n.Layers {
		if mem[i][3] > 0 {
			s += fmt.Sprintf("%2d: %-30s %8s %8s %8s %8s\n", i, layer.ToString(), FormatBytes(mem[i][0]),
				FormatBytes(mem[i][1]), FormatBytes(mem[i][2]+n.workSize[i]), FormatBytes(mem[i][3]+n.workSize[i]))
		}
	}
	s += fmt.Sprintf("--- TOTAL --- %20s %8s %8s %8s %8s\n", "", FormatBytes(total[0]), FormatBytes(total[1]),
		FormatBytes(total[2]), FormatBytes(total[3]))
	return s
}

func (n *Network) meminfo() (mem [][4]int, total [4]int) {
	var r [4]int
	for _, layer := range n.Layers {
		r[0], r[1], r[2] = layer.Memory()
		r[3] = r[0] + r[1] + r[2]
		for i, val := range r {
			total[i] += val
		}
		mem = append(mem, r)
	}
	totalWork := max(n.workSize)
	total[2] += totalWork
	total[3] += totalWork
	return
}

// Print network weights
func (n *Network) PrintWeights() {
	for i, layer := range n.Layers {
		if l, ok := layer.(ParamLayer); ok {
			W, B, _, _ := l.Params()
			log.Printf("== Layer %d weights ==\n%s %s\n", i, W.String(n.queue), B.String(n.queue))
		}
	}
}

func (n *Network) allocArrays(size int) {
	n.classes = n.queue.NewArray(num.Int32, size)
	n.diffs = n.queue.NewArray(num.Int32, size)
	n.batchLoss = n.queue.NewArray(num.Float32)
	n.totalLoss = n.queue.NewArray(num.Float32)
	n.batchErr = n.queue.NewArray(num.Float32)
	n.totalErr = n.queue.NewArray(num.Float32)
}

// Set random number seed, or random seed if seed <= 0
func SetSeed(seed int64) *rand.Rand {
	if seed <= 0 {
		seed = time.Now().UTC().UnixNano()
	}
	log.Println("random seed =", seed)
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

// Convert no. of bytes to string
func FormatBytes(n int) string {
	if n <= 0 {
		return ""
	}
	if n < 1024*1024 {
		return fmt.Sprintf("%dK", 1+(n-1)/1024)
	}
	return fmt.Sprintf("%dM", 1+(n-1)/(1024*1024))
}

func max(arr []int) int {
	m := arr[0]
	if len(arr) > 1 {
		for _, val := range arr[1:] {
			if val > m {
				m = val
			}
		}
	}
	return m
}
