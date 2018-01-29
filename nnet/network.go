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
	Layers     []Layer
	WorkSpace  num.Buffer
	queue      num.Queue
	classes    *num.Array
	diffs      *num.Array
	batchErr   *num.Array
	totalErr   *num.Array
	batchLoss  *num.Array
	totalLoss  *num.Array
	bpropPool1 num.Buffer
	bpropPool2 num.Buffer
	inShape    []int
	workSize   []int
	inputSize  []int
}

// New function creates a new network with the given layers. If bprop is true then allocate memory for back propagation.
func New(queue num.Queue, conf Config, batchSize int, inShape []int, bprop bool, rng *rand.Rand) *Network {
	n := &Network{Config: conf, queue: queue}
	n.allocArrays(batchSize)
	n.inShape = append(inShape, batchSize)
	shape := n.inShape
	n.workSize = make([]int, len(conf.Layers)+1)
	n.inputSize = make([]int, len(conf.Layers)+1)
	opts := num.FpropOnly
	if bprop {
		opts |= num.BpropWeights
	}
	if conf.Momentum != 0 {
		opts |= num.MomentumUpdate
	}
	weightSize := 0
	for i, l := range conf.Layers {
		layer := l.Unmarshal()
		n.workSize[i] = layer.Init(queue, shape, opts, rng)
		n.Layers = append(n.Layers, layer)
		if opts&num.BpropData != 0 {
			n.inputSize[i] = num.Prod(layer.InShape())
		}
		if conf.DebugLevel >= 1 {
			log.Printf("init layer %d: %s opts=%s work=%d insize=%d\n", i, l.Type, opts, n.workSize[i], n.inputSize[i])
		}
		shape = layer.OutShape()
		if l.Type != "flatten" && bprop {
			opts |= num.BpropData
		}
		if lp, ok := layer.(ParamLayer); ok && bprop && conf.Nesterov {
			size := num.Prod(lp.FilterShape())
			if lp.BiasShape() != nil {
				size += num.Prod(lp.BiasShape())
			}
			if size > weightSize {
				weightSize = size
			}
		}
	}
	if opts&num.BpropData != 0 {
		n.inputSize[len(n.Layers)] = num.Prod(shape)
	}
	n.workSize[len(n.Layers)] = weightSize
	wsize := max(n.workSize...)
	insize := max(n.inputSize...)
	if conf.DebugLevel >= 1 {
		log.Printf("maxWorkSize=%d  maxInSize=%d  maxWeightSize=%d\n", wsize, insize, weightSize)
	}
	if wsize > 0 {
		n.WorkSpace = queue.NewBuffer(wsize)
	}
	if insize > 0 {
		n.bpropPool1 = queue.NewBuffer(insize)
		n.bpropPool2 = queue.NewBuffer(insize)
	}
	return n
}

// release allocated buffers
func (n *Network) Release() {
	n.queue.Finish()
	for _, layer := range n.Layers {
		layer.Release()
	}
	num.Release(n.classes, n.diffs, n.batchLoss, n.batchErr, n.totalLoss, n.totalErr,
		n.WorkSpace, n.bpropPool1, n.bpropPool2)
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
func (n *Network) Fprop(input *num.Array, trainMode bool) *num.Array {
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
func (n *Network) Bprop(batch int, yPred, yOneHot *num.Array, opt Optimiser) {
	q := n.queue
	pool1, pool2 := n.bpropPool1, n.bpropPool2
	grad := num.NewArray(pool1, num.Float32, yOneHot.Dims...)
	q.Call(
		num.Copy(yPred, grad),
		num.Axpy(-1, yOneHot, grad),
	)
	if n.DebugLevel >= 2 || (n.DebugLevel == 1 && batch == 0) {
		log.Printf("input grad:\n%s", grad.String(q))
	}
	for i := len(n.Layers) - 1; i >= 0; i-- {
		layer := n.Layers[i]
		var dsrc *num.Array
		if n.inputSize[i] > 0 {
			dsrc = num.NewArray(pool2, num.Float32, layer.InShape()...)
		}
		grad = layer.Bprop(q, grad, dsrc, n.WorkSpace)
		if n.DebugLevel >= 3 && grad != nil {
			log.Printf("layer %d bprop output:\n%s", i, grad.String(q))
		}
		pool1, pool2 = pool2, pool1
	}
	for _, layer := range n.Layers {
		if l, ok := layer.(ParamLayer); ok {
			W, B := l.Params()
			dW, dB := l.ParamGrads()
			vW, vB := l.ParamVelocity()
			opt.Update(q, l.Type() != "batchNorm", W, dW, vW, n.WorkSpace)
			if B != nil {
				opt.Update(q, false, B, dB, vB, n.WorkSpace)
			}
		}
	}
}

// get the loss for this batch
func (n *Network) calcBatchLoss(batch int, yOneHot, yPred *num.Array) {
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
func (n *Network) calcBatchError(batch int, dset *Dataset, y, yPred *num.Array, pred []int32) {
	q := n.queue
	q.Call(
		num.Unhot(yPred, n.classes),
		num.Neq(n.classes, y, n.diffs),
		num.Sum(n.diffs, n.batchErr),
		num.Axpy(1, n.batchErr, n.totalErr),
	)
	if n.DebugLevel >= 2 || (n.DebugLevel >= 1 && batch == 0) {
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
			input := n.inputSize[i] * 4
			work := n.workSize[i] * 4
			s += fmt.Sprintf("%2d: %-30s %8s %8s %8s %8s\n", i, layer.ToString(),
				FormatBytes(mem[i][0]), FormatBytes(mem[i][1]+input), FormatBytes(mem[i][2]+work),
				FormatBytes(mem[i][3]+input+work))
		}
	}
	s += fmt.Sprintf("--- TOTAL --- %20s %8s %8s %8s %8s\n", "",
		FormatBytes(total[0]), FormatBytes(total[1]), FormatBytes(total[2]), FormatBytes(total[3]))
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
	totalInput := max(n.inputSize...) * 8
	totalWork := max(n.workSize...) * 4
	total[1] += totalInput
	total[2] += totalWork
	total[3] += totalInput + totalWork
	return
}

// Print network weights
func (n *Network) PrintWeights() {
	for i, layer := range n.Layers {
		if l, ok := layer.(ParamLayer); ok {
			W, B := l.Params()
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

func max(arr ...int) int {
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
