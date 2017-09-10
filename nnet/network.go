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
}

// New function creates a new network with the given layers.
func New(conf Config) *Network {
	n := &Network{Config: conf}
	for _, l := range conf.Layers {
		n.Layers = append(n.Layers, l.Unmarshal())
	}
	return n
}

// Accessor for output layer
func (n *Network) OutLayer() OutputLayer {
	return n.Layers[len(n.Layers)-1].(OutputLayer)
}

// Initialise network weights using a linear or normal distribution.
// Weights for each layer are scaled by 1/sqrt(nin)
func (n *Network) InitWeights(q num.Queue, inShape []int) {
	shape := inShape
	if n.FlattenInput {
		shape = []int{num.Prod(inShape)}
	}
	for _, layer := range n.Layers {
		if l, ok := layer.(ParamLayer); ok {
			scale := float32(1 / math.Sqrt(float64(shape[0])))
			l.InitParams(q, shape, scale, n.NormalWeights)
		}
		shape = layer.OutShape(shape)
	}
	if n.DebugLevel >= 2 {
		n.PrintWeights(q)
	}
}

// Feed forward the input to get the predicted output
func (n *Network) Fprop(q num.Queue, input num.Array) num.Array {
	pred := input
	for i, layer := range n.Layers {
		if n.DebugLevel >= 3 {
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
		if n.DebugLevel >= 2 {
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
	for i, layer := range n.Layers {
		s[i] = fmt.Sprintf("%2d: %s", i, layer)
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
		n.classes = num.NewArray(q.Device(), num.Int32, size)
		n.diffs = num.NewArray(q.Device(), num.Int32, size)
		n.batchErr = num.NewArray(q.Device(), num.Float32)
		n.total = num.NewArray(q.Device(), num.Float32)
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
