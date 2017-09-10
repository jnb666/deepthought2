package nnet

import (
	"fmt"
	"github.com/jnb666/deepthought2/num"
	"time"
)

const statsBufferSize = 10

// Training statistics
type Stats struct {
	Epoch   int
	Values  []float64
	Elapsed time.Duration
}

func StatsHeaders(d map[string]Data) []string {
	h := []string{"loss"}
	for _, key := range DataTypes {
		if _, ok := d[key]; ok {
			h = append(h, key+" error")
		}

	}
	return h
}

func (s Stats) Format() []string {
	str := []string{fmt.Sprintf("%7.4f", s.Values[0])}
	for _, v := range s.Values[1:] {
		str = append(str, fmt.Sprintf("%6.2f%%", v*100))
	}
	return str
}

// Tester interface to evaluate the performance after each epoch, Test method returns true if training should stop.
type Tester interface {
	Test(q num.Queue, net *Network, epoch int, loss float64, start time.Time) bool
}

// Tester which evaluates the loss and error for each of the data sets and updates the stats.
type TestBase struct {
	Data    map[string]*Dataset
	Stats   Stats
	Headers []string
}

// Create a new base class which implements the Tester interface.
func NewTestBase(device num.Device, data map[string]Data, conf Config) *TestBase {
	t := &TestBase{Data: make(map[string]*Dataset), Headers: StatsHeaders(data)}
	for key, d := range data {
		t.Data[key] = NewDataset(device, d, conf.TestBatch, conf.MaxSamples)
	}
	return t
}

// Test performance of the network, called from the Train function on completion of each epoch.
func (t *TestBase) Test(q num.Queue, net *Network, epoch int, loss float64, start time.Time) bool {
	t.Stats.Epoch = epoch
	t.Stats.Values = []float64{loss}
	for _, key := range DataTypes {
		if _, ok := t.Data[key]; ok {
			t.Stats.Values = append(t.Stats.Values, net.Error(q, t.Data[key], nil))
		}
	}
	t.Stats.Elapsed = time.Since(start)
	return epoch >= net.MaxEpoch || loss <= net.MinLoss
}

type testLogger struct {
	*TestBase
}

// Create a new tester which logs stats to stdout.
func NewTestLogger(device num.Device, data map[string]Data, conf Config) Tester {
	return testLogger{TestBase: NewTestBase(device, data, conf)}
}

func (t testLogger) Test(q num.Queue, net *Network, epoch int, loss float64, start time.Time) bool {
	done := t.TestBase.Test(q, net, epoch, loss, start)
	if done || net.LogEvery == 0 || epoch%net.LogEvery == 0 {
		msg := fmt.Sprintf("epoch %3d:", epoch)
		for i, val := range t.Stats.Format() {
			msg += fmt.Sprintf("  %s =%s", t.Headers[i], val)
		}
		fmt.Println(msg)
	}
	if done {
		fmt.Printf("run time: %.2gs\n", t.Stats.Elapsed.Seconds())
	}
	return done
}

// Train the network on the given training set by updating the weights
func Train(q num.Queue, net *Network, data Data, test Tester) {
	dset := NewDataset(q.Device(), data, net.TrainBatch, net.MaxSamples)
	acc := num.NewArray(q.Device(), num.Float32)
	epoch := 1
	done := false
	start := time.Now()
	for !done {
		loss := TrainEpoch(q, net, dset, acc)
		done = test.Test(q, net, epoch, loss, start)
		epoch++
	}
}

// Perform one training epoch on dataset, returns the current loss prior to updating the weights.
func TrainEpoch(q num.Queue, net *Network, dset *Dataset, acc num.Array) float64 {
	if net.Shuffle {
		dset.Shuffle()
	}
	nbatch := dset.Batches()
	weightDecay := float32(net.Eta*net.Lambda) / float32(dset.Samples)
	q.Call(num.Fill(acc, 0))
	if net.inputGrad == nil || net.inputGrad.Dims()[0] != dset.BatchSize {
		net.inputGrad = num.NewArray(q.Device(), num.Float32, dset.BatchSize, dset.Classes)
		net.batchLoss = num.NewArray(q.Device(), num.Float32)
	}
	for batch := 0; batch < nbatch; batch++ {
		if net.DebugLevel >= 1 {
			fmt.Printf("== train batch %d ==\n", batch)
		}
		x, _, yOneHot := dset.GetBatch(q, batch)
		yPred := net.Fprop(q, x)
		if net.DebugLevel >= 2 {
			fmt.Printf("yOneHot:\n%s", yOneHot.String(q))
			fmt.Printf("yPred:\n%s", yPred.String(q))
		}
		// sum average loss over batches
		losses := net.OutLayer().Loss(q, yOneHot, yPred)
		q.Call(
			num.Sum(losses, net.batchLoss, 1),
			num.Axpy(1/float32(dset.Samples), net.batchLoss, acc),
		)
		// get difference at output
		q.Call(
			num.Copy(net.inputGrad, yPred),
			num.Axpy(-1, yOneHot, net.inputGrad),
		)
		if net.DebugLevel >= 1 {
			fmt.Printf("input grad:\n%s", net.inputGrad.String(q))
		}
		grad := net.inputGrad
		// back propagate gradient
		for i := len(net.Layers) - 1; i >= 0; i-- {
			layer := net.Layers[i]
			grad = layer.Bprop(q, grad)
			if net.DebugLevel >= 2 || (net.DebugLevel == 1 && i == 0) {
				fmt.Printf("layer %d bprop output:\n%s", i, grad.String(q))
			}
		}
		// update weights
		for ix, layer := range net.Layers {
			if l, ok := layer.(ParamLayer); ok {
				W, B := l.Params()
				dW, dB := l.ParamGrads()
				if weightDecay != 0 {
					q.Call(num.Axpy(-weightDecay, W, dW))
				}
				q.Call(
					num.Axpy(-float32(net.Eta), dW, W),
					num.Axpy(-float32(net.Eta), dB, B),
				).Finish()
				if net.DebugLevel >= 1 {
					fmt.Printf("layer %d weights\n%s", ix, W.String(q))
					fmt.Printf("layer %d bias\n %s\n %s\n", ix, dB.String(q), B.String(q))
				}
			}
		}
	}
	lossVal := make([]float32, 1)
	q.Call(num.Read(acc, lossVal)).Finish()
	return float64(lossVal[0])
}
