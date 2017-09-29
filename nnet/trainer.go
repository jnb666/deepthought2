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
	Test(net *Network, epoch int, loss float64, start time.Time) bool
}

// Tester which evaluates the loss and error for each of the data sets and updates the stats.
type TestBase struct {
	Net     *Network
	Data    map[string]*Dataset
	Predict map[string][]int32
	Stats   Stats
	Headers []string
	Samples int
}

// Create a new base class which implements the Tester interface.
// If limitSamples flag is set then total no. of samples will be entries in smallest dataset.
func NewTestBase(queue num.Queue, conf Config, data map[string]Data, limitSamples bool) *TestBase {
	t := &TestBase{
		Data:    make(map[string]*Dataset),
		Headers: StatsHeaders(data),
		Samples: min(conf.MaxSamples, len(data["train"].Labels)),
	}
	if limitSamples {
		for _, d := range data {
			if len(d.Labels) < t.Samples {
				t.Samples = len(d.Labels)
			}
		}
	}
	if conf.DebugLevel >= 1 {
		fmt.Printf("init tester: samples=%d batch size=%d\n", t.Samples, conf.TestBatch)
	}
	for key, d := range data {
		if conf.DebugLevel >= 1 {
			fmt.Println("dataset =>", key)
		}
		t.Data[key] = NewDataset(queue.Dev(), d, conf.TestBatch, t.Samples)
	}
	t.Net = New(queue, conf, t.Data["train"].BatchSize, t.Data["train"].Shape)
	return t
}

// Test performance of the network, called from the Train function on completion of each epoch.
// If Predict map is not nil then save predicted results.
func (t *TestBase) Test(net *Network, epoch int, loss float64, start time.Time) bool {
	net.CopyTo(t.Net)
	if net.DebugLevel >= 1 {
		fmt.Printf("== TEST EPOCH %d ==\n", epoch)
	}
	t.Stats.Epoch = epoch
	t.Stats.Values = []float64{loss}
	for _, key := range DataTypes {
		if dset, ok := t.Data[key]; ok {
			if dset.Samples < len(dset.Labels) {
				dset.Shuffle(t.Net.rng)
			}
			var pred []int32
			if t.Predict != nil {
				pred = t.Predict[key]
			}
			t.Stats.Values = append(t.Stats.Values, t.Net.Error(dset, pred))
		}
	}
	t.Stats.Elapsed = time.Since(start)
	return epoch >= net.MaxEpoch || loss <= net.MinLoss
}

type testLogger struct {
	*TestBase
}

// Create a new tester which logs stats to stdout.
func NewTestLogger(queue num.Queue, conf Config, data map[string]Data) Tester {
	return testLogger{TestBase: NewTestBase(queue, conf, data, true)}
}

func (t testLogger) Test(net *Network, epoch int, loss float64, start time.Time) bool {
	done := t.TestBase.Test(net, epoch, loss, start)
	if done || net.LogEvery == 0 || epoch%net.LogEvery == 0 {
		msg := fmt.Sprintf("epoch %3d:", epoch)
		for i, val := range t.Stats.Format() {
			msg += fmt.Sprintf("  %s =%s", t.Headers[i], val)
		}
		fmt.Println(msg)
	}
	if done {
		fmt.Printf("run time: %s\n", t.Stats.Elapsed.Round(10*time.Millisecond))
	}
	return done
}

// Train the network on the given training set by updating the weights
func Train(net *Network, dset *Dataset, test Tester) {
	acc := net.queue.NewArray(num.Float32)
	done := false
	start := time.Now()
	for epoch := 1; epoch <= net.MaxEpoch && !done; epoch++ {
		loss := TrainEpoch(net, dset, acc)
		done = test.Test(net, epoch, loss, start)
	}
}

// Perform one training epoch on dataset, returns the current loss prior to updating the weights.
func TrainEpoch(net *Network, dset *Dataset, acc num.Array) float64 {
	q := net.queue
	if net.inputGrad == nil {
		net.inputGrad = q.NewArray(num.Float32, dset.Classes, dset.BatchSize)
	}
	if net.Shuffle {
		dset.Shuffle(net.rng)
	}
	weightDecay := float32(net.Eta*net.Lambda) / float32(dset.Samples)
	q.Call(num.Fill(acc, 0))
	dset.Rewind()
	for batch := 0; batch < dset.Batches; batch++ {
		if net.DebugLevel >= 2 || (net.DebugLevel == 1 && batch == 0) {
			fmt.Printf("== train batch %d ==\n", batch)
		}
		q.Finish()
		x, _, yOneHot := dset.NextBatch()
		yPred := net.Fprop(x)
		if net.DebugLevel >= 2 {
			fmt.Printf("yOneHot:\n%s", yOneHot.String(q))
			fmt.Printf("yPred:\n%s", yPred.String(q))
		}
		// sum average loss over batches
		losses := net.OutLayer().Loss(q, yOneHot, yPred)
		if net.DebugLevel >= 2 {
			fmt.Printf("loss:\n%s", losses.String(q))
		}
		q.Call(
			num.Sum(losses, net.batchLoss),
			num.Axpy(1, net.batchLoss, acc),
		)
		// get difference at output
		q.Call(
			num.Copy(yPred, net.inputGrad),
			num.Axpy(-1, yOneHot, net.inputGrad),
		)
		if net.DebugLevel >= 2 || (net.DebugLevel == 1 && batch == 0) {
			fmt.Printf("input grad:\n%s", net.inputGrad.String(q))
		}
		grad := net.inputGrad
		// back propagate gradient
		for i := len(net.Layers) - 1; i >= 0; i-- {
			layer := net.Layers[i]
			grad = layer.Bprop(q, grad, net.WorkSpace)
			if net.DebugLevel >= 3 && grad != nil {
				fmt.Printf("layer %d bprop output:\n%s", i, grad.String(q))
			}
		}
		// update weights
		for _, layer := range net.Layers {
			if l, ok := layer.(ParamLayer); ok {
				l.UpdateParams(q, float32(net.Eta), weightDecay)
			}
		}
		if net.DebugLevel >= 2 || (batch == dset.Batches-1 && net.DebugLevel >= 1) {
			net.PrintWeights()
		}
	}
	lossVal := make([]float32, 1)
	q.Call(num.Read(acc, lossVal)).Finish()
	return float64(lossVal[0] / float32(dset.Samples))
}

func min(a, b int) int {
	if a == 0 {
		return b
	}
	if a < b {
		return a
	}
	return b
}
