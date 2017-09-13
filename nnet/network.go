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

// Network type represents a multilayer neural network model.
type Network struct {
	Config
	Layers    []Layer
	classes   num.Array
	diffs     num.Array
	total     num.Array
	batchErr  num.Array
	batchLoss num.Array
	inputGrad num.Array
	inShape   []int
}

// New function creates a new network with the given layers.
func New(dev num.Device, conf Config, batchSize int, inShape []int) *Network {
	n := &Network{Config: conf}
	if conf.FlattenInput {
		n.inShape = []int{batchSize, num.Prod(inShape)}
	} else {
		n.inShape = append([]int{batchSize}, inShape...)
	}
	shape := n.inShape
	var prev Layer
	for _, l := range conf.Layers {
		layer := l.Unmarshal()
		layer.Init(dev, shape, prev)
		n.Layers = append(n.Layers, layer)
		shape = layer.OutShape(shape)
		prev = layer
	}
	// add backward links for DNN layers
	var next Layer
	for i := len(n.Layers) - 1; i >= 0; i-- {
		if l, ok := n.Layers[i].(DNNLayer); ok {
			l.Link(dev, next)
			if conf.DebugLevel >= 1 {
				fmt.Printf("== DNN layer %d ==\n%s", i, l.Get())
			}
		}
		next = n.Layers[i]
	}
	return n
}

// Initialise network weights using a linear or normal distribution.
// Weights for each layer are scaled by 1/sqrt(nin)
func (n *Network) InitWeights(q num.Queue) {
	shape := n.inShape
	for _, layer := range n.Layers {
		if l, ok := layer.(ParamLayer); ok {
			nin := num.Prod(shape[1:])
			scale := float32(1 / math.Sqrt(float64(nin)))
			l.InitParams(q, scale, n.NormalWeights)
		}
		shape = layer.OutShape(shape)
	}
	if n.DebugLevel >= 2 {
		n.PrintWeights(q)
	}
}

// Copy weights and bias arrays to destination net
func (n *Network) CopyTo(q num.Queue, net *Network) {
	for i, layer := range n.Layers {
		if l, ok := layer.(ParamLayer); ok {
			W, B := l.Params()
			net.Layers[i].(ParamLayer).SetParams(q, W, B)
		}
	}
}

// Accessor for output layer
func (n *Network) OutLayer() OutputLayer {
	return n.Layers[len(n.Layers)-1].(OutputLayer)
}

// Feed forward the input to get the predicted output
func (n *Network) Fprop(q num.Queue, input num.Array) num.Array {
	pred := input
	for i, layer := range n.Layers {
		if n.DebugLevel >= 2 && pred != nil {
			fmt.Printf("layer %d input\n%s", i, pred.String(q))
		}
		pred = layer.Fprop(q, pred)
	}
	return pred
}

// Predict output given input data
func (n *Network) Predict(q num.Queue, input, classes num.Array) num.Array {
	yPred := n.Fprop(q, input)
	if n.DebugLevel >= 2 {
		fmt.Printf("yPred\n%s", yPred.String(q))
	}
	q.Call(num.Unhot(yPred, classes))
	return yPred
}

// Calculate the error from the predicted versus actual values
// if pred slice is not nil then also return the predicted output classes.
func (n *Network) Error(q num.Queue, dset *Dataset, pred []int32) float64 {
	n.allocArrays(q, dset.BatchSize)
	q.Call(num.Fill(n.total, 0))
	nbatch := dset.Batches()
	for batch := 0; batch < nbatch; batch++ {
		x, y, _ := dset.GetBatch(q, batch)
		n.Predict(q, x, n.classes)
		q.Call(
			num.Neq(n.classes, y, n.diffs),
			num.Sum(n.diffs, n.batchErr, 1),
			num.Axpy(1, n.batchErr, n.total),
		)
		if pred != nil {
			start := batch * dset.BatchSize
			end := start + y.Dims()[0]
			q.Call(num.Read(n.classes, pred[start:end]))
		}
		if n.DebugLevel >= 2 || (n.DebugLevel >= 1 && batch == 0) {
			fmt.Printf("batch %d error =%s\n", batch, n.batchErr.String(q))
			fmt.Println(y.String(q))
			fmt.Println(n.classes.String(q))
		}
	}
	err := []float32{0}
	q.Call(num.Read(n.total, err)).Finish()
	return float64(err[0]) / float64(dset.Samples)
}

// Print network description
func (n *Network) String() string {
	s := make([]string, len(n.Layers))
	shape := n.inShape
	for i, layer := range n.Layers {
		s[i] = fmt.Sprintf("%2d: %-25s %v", i, layer, shape)
		shape = layer.OutShape(shape)
	}
	return fmt.Sprintf("== Config ==\n%s\n== Network ==\n%s", n.Config, strings.Join(s, "\n"))
}

// Print network weights
func (n *Network) PrintWeights(q num.Queue) {
	for i, layer := range n.Layers {
		if l, ok := layer.(ParamLayer); ok {
			W, B := l.Params()
			fmt.Printf("== Layer %d weights ==\n%s %s\n", i, W.String(q), B.String(q))
		}
	}
}

func (n *Network) allocArrays(q num.Queue, size int) {
	if n.classes == nil || n.classes.Dims()[0] != size {
		n.classes = q.NewArray(num.Int32, size)
		n.diffs = q.NewArray(num.Int32, size)
		n.batchErr = q.NewArray(num.Float32)
		n.total = q.NewArray(num.Float32)
	}
}

// Set random number seed, or random seed id seed <= 0
func SetSeed(seed int64) {
	if seed <= 0 {
		seed = time.Now().UTC().UnixNano()
	}
	fmt.Println("random seed =", seed)
	rand.Seed(seed)
}

// Exit in case of error
func CheckErr(err error) {
	if err != nil {
		fmt.Fprintln(os.Stderr, err)
		os.Exit(1)
	}
}
