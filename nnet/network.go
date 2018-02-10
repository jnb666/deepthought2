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

var debug int

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
	InShape   []int
	WorkSpace [3]num.Buffer
	queue     num.Queue
	classes   *num.Array
	diffs     *num.Array
	total     *num.Array
	workSize  []int
	inputSize []int
}

// New function creates a new network with the given layers. If bprop is true then allocate memory for back propagation.
func New(queue num.Queue, conf Config, batchSize int, inShape []int, bprop bool, rng *rand.Rand) *Network {
	debug = conf.DebugLevel
	n := &Network{Config: conf, queue: queue}
	n.allocArrays(batchSize)
	n.InShape = append(inShape, batchSize)
	shape := n.InShape
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
		if debug >= 1 {
			log.Printf("init layer %d: %s %v => %v opts=%s work=%d insize=%d\n",
				i, l.Type, shape, layer.OutShape(), opts, n.workSize[i], n.inputSize[i])
		}
		shape = layer.OutShape()
		if l.Type != "flatten" && bprop {
			opts |= num.BpropData
		}
		if lp, ok := layer.(ParamLayer); ok && bprop && conf.Nesterov {
			if size := lp.NumWeights(); size > weightSize {
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
	if debug >= 1 {
		log.Printf("maxWorkSize=%d  maxInSize=%d  maxWeightSize=%d\n", wsize, insize, weightSize)
	}
	if wsize > 0 {
		n.WorkSpace[0] = queue.NewBuffer(wsize)
	}
	if insize > 0 {
		n.WorkSpace[1] = queue.NewBuffer(insize)
		n.WorkSpace[2] = queue.NewBuffer(insize)
	}
	return n
}

// release allocated buffers
func (n *Network) Release() {
	n.queue.Finish()
	for _, layer := range n.Layers {
		layer.Release()
	}
	num.Release(n.classes, n.diffs, n.total, n.WorkSpace[0], n.WorkSpace[1], n.WorkSpace[2])
}

// Initialise network weights using a linear or normal distribution.
// Weights for each layer are scaled by 1/sqrt(nin)
func (n *Network) InitWeights(rng *rand.Rand) {
	ParamLayers("", n.Layers, func(desc string, l ParamLayer) {
		l.InitParams(n.queue, n.WeightInit, n.Bias, rng)
	})
	if debug >= 2 {
		n.PrintWeights()
	}
}

// Accessor for output layer
func (n *Network) OutLayer() OutputLayer {
	return n.Layers[len(n.Layers)-1].(OutputLayer)
}

// get the average loss for this batch
func (n *Network) BatchLoss(yOneHot, yPred *num.Array) float64 {
	q := n.queue
	losses := n.OutLayer().Loss(q, yOneHot, yPred)
	if debug >= 2 {
		log.Printf("loss:\n%s", losses.String(q))
	}
	var total = []float32{0}
	q.Call(
		num.Sum(losses, n.total),
		num.Read(n.total, total),
	).Finish()
	return float64(total[0]) / float64(yOneHot.Dims[1])
}

// get the total error for this batch, if pred is non-null then save predicted values
func (n *Network) BatchError(batch int, dset *Dataset, y, yPred *num.Array, pred []int32) float64 {
	q := n.queue
	q.Call(
		num.Unhot(yPred, n.classes),
		num.Neq(n.classes, y, n.diffs),
		num.Sum(n.diffs, n.total),
	)
	if debug >= 2 || (debug == 1 && batch == 0) {
		log.Printf("batch error = %s\n", n.total.String(q))
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
	var total = []float32{0}
	q.Call(num.Read(n.total, total)).Finish()
	return float64(total[0])
}

// Calculate the error from the predicted versus actual values
// if pred slice is not nil then also return the predicted output classes.
func (n *Network) Error(dset *Dataset, pred []int32) float64 {
	dset.NextEpoch()
	var p []int32
	if pred != nil {
		p = make([]int32, dset.Samples)
	}
	totalErr := 0.0
	for batch := 0; batch < dset.Batches; batch++ {
		n.queue.Finish()
		x, y, _ := dset.NextBatch()
		yPred := Fprop(n.queue, n.Layers, x, n.WorkSpace[0], false)
		totalErr += n.BatchError(batch, dset, y, yPred, p)
	}
	if pred != nil {
		for i, ix := range dset.indexes {
			pred[ix] = p[i]
		}
	}
	return totalErr / float64(dset.Samples)
}

// Print network description
func (n *Network) String() string {
	s := n.configString()
	if n.Layers == nil {
		return s
	}
	s += fmt.Sprintf("\n== Network ==\n    %-12s input", fmt.Sprint(n.InShape[:len(n.InShape)-1]))
	for i, layer := range n.Layers {
		s += fmt.Sprintf("\n%2d: %s", i, layer.ToString())
		if group, ok := layer.(LayerGroup); ok {
			desc := group.LayerDesc()
			for j, l := range group.Layers() {
				s += fmt.Sprintf("\n     %s %s", desc[j], l.ToString())
			}
		}
	}
	totalWeights := 0
	ParamLayers("", n.Layers, func(desc string, l ParamLayer) {
		totalWeights += l.NumWeights()
	})
	return s + fmt.Sprintf("\ntotal weights %d", totalWeights)
}

// Get total allocated memory in bytes
func (n *Network) Memory() int {
	m := new(memInfo)
	m.update(n.Layers, -1, n.inputSize, n.workSize)
	return m.total[3]
}

// Print profile of allocated memory
func (n *Network) MemoryProfile() string {
	m := new(memInfo)
	m.update(n.Layers, -1, n.inputSize, n.workSize)
	s := fmt.Sprintf("== memory profile ==                weights  outputs     temp    total (%d)\n",
		n.InShape[len(n.InShape)-1])
	for i, mem := range m.bytes {
		if mem[3] > 0 {
			s += fmt.Sprintf("%-34s %8s %8s %8s %8s\n", m.name[i],
				FormatBytes(mem[0]), FormatBytes(mem[1]), FormatBytes(mem[2]), FormatBytes(mem[3]))
		}
	}
	s += fmt.Sprintf("--- TOTAL --- %20s %8s %8s %8s %8s\n", "",
		FormatBytes(m.total[0]), FormatBytes(m.total[1]), FormatBytes(m.total[2]), FormatBytes(m.total[3]))
	return s
}

// Print network weights
func (n *Network) PrintWeights() {
	ParamLayers("", n.Layers, func(desc string, l ParamLayer) {
		W, B := l.Params()
		log.Printf("== Layer %s weights ==\n%s", desc, W.String(n.queue))
		if B != nil {
			log.Printf(" %s", B.String(n.queue))
		}
		log.Println()
	})
}

func (n *Network) allocArrays(size int) {
	n.classes = n.queue.NewArray(num.Int32, size)
	n.diffs = n.queue.NewArray(num.Int32, size)
	n.total = n.queue.NewArray(num.Float32)
}

type memInfo struct {
	name  []string
	bytes [][4]int
	total [4]int
}

func (m *memInfo) update(layers []Layer, layerId int, inputSize, workSize []int) {
	var r [4]int
	for i, layer := range layers {
		r[0], r[1], r[2] = layer.Memory()
		r[3] = r[0] + r[1] + r[2]
		for i, val := range r {
			m.total[i] += val
		}
		if layerId < 0 {
			input := inputSize[i] * 4
			work := workSize[i] * 4
			r[1] += input
			r[2] += work
			r[3] += input + work
			m.name = append(m.name, fmt.Sprintf("%2d: %s", i, layer))
		} else {
			m.name = append(m.name, fmt.Sprintf("      %s", layer))
		}
		m.bytes = append(m.bytes, r)
		if l, ok := layer.(LayerGroup); ok {
			m.update(l.Layers(), i, inputSize, workSize)
		}
	}
	if layerId < 0 {
		totalInput := max(inputSize...) * 8
		totalWork := max(workSize...) * 4
		m.total[1] += totalInput
		m.total[2] += totalWork
		m.total[3] += totalInput + totalWork
	}
}

// Call function on each of the ParamLayers in the network
func ParamLayers(desc string, layers []Layer, callback func(desc string, l ParamLayer)) {
	for i, layer := range layers {
		if l, ok := layer.(ParamLayer); ok {
			callback(fmt.Sprintf("%s%d", desc, i), l)
		}
		if l, ok := layer.(LayerGroup); ok {
			ParamLayers(fmt.Sprintf("%d.", i), l.Layers(), callback)
		}
	}
}

// Copy weights and bias arrays from src to dst
func CopyParams(q num.Queue, src, dst []Layer, sync bool) {
	for i, layer := range dst {
		if l, ok := layer.(ParamLayer); ok {
			l.Copy(q, src[i])
		}
		if l, ok := layer.(LayerGroup); ok {
			CopyParams(q, src[i].(LayerGroup).Layers(), l.Layers(), false)
		}
	}
	if sync {
		q.Finish()
	}
}

// Feed forward the input to get the predicted output
func Fprop(q num.Queue, layers []Layer, input *num.Array, work num.Buffer, trainMode bool) *num.Array {
	pred := input
	for i, layer := range layers {
		if debug >= 2 && pred != nil {
			log.Printf("layer %d input\n%s", i, pred.String(q))
		}
		pred = layer.Fprop(q, pred, work, trainMode)
	}
	return pred
}

// Back propagate gradient through the layers
func Bprop(q num.Queue, layers []Layer, grad *num.Array, work [3]num.Buffer) *num.Array {
	tmp1, tmp2 := work[1], work[2]
	for i := len(layers) - 1; i >= 0; i-- {
		layer := layers[i]
		var dsrc *num.Array
		if layer.BpropData() {
			dsrc = num.NewArray(tmp2, num.Float32, layer.InShape()...)
		}
		grad = layer.Bprop(q, grad, dsrc, work)
		if debug >= 3 && grad != nil {
			log.Printf("layer %d bprop output:\n%s", i, grad.String(q))
		}
		tmp1, tmp2 = tmp2, tmp1
	}
	return grad
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
