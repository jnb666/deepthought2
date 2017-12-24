package nnet

import (
	"fmt"
	"github.com/jnb666/deepthought2/num"
	"html/template"
	"math"
	"math/rand"
	"time"
)

const statsBufferSize = 10

const (
	emaN = 10
	emaK = 2.0 / (emaN + 1.0)
)

// Training statistics
type Stats struct {
	Epoch     int
	Values    []float64
	BestSince int
	Elapsed   time.Duration
}

func StatsHeaders(d map[string]Data) []string {
	h := []string{"loss"}
	for _, key := range DataTypes {
		if _, ok := d[key]; ok {
			h = append(h, key+" error")
			if key == "valid" {
				h = append(h, "valid avg")
			}
		}
	}
	return h
}

func (s Stats) Copy() Stats {
	stats := s
	stats.Values = append([]float64{}, s.Values...)
	return stats
}

func (s Stats) Format() []string {
	str := make([]string, len(s.Values))
	for i := range str {
		str[i] = s.FormatItem(i)
	}
	return str
}

func (s Stats) FormatItem(i int) string {
	if i == 0 {
		return fmt.Sprintf("%7.4f", s.Values[0])
	}
	return fmt.Sprintf("%6.2f%%", s.Values[i]*100)
}

func (s Stats) String(headers []string, done bool) string {
	msg := fmt.Sprintf("epoch %3d:", s.Epoch)
	for i, val := range s.Format() {
		msg += fmt.Sprintf("  %s =%s", headers[i], val)
	}
	if done {
		msg += fmt.Sprintf("\nrun time: %s", s.Elapsed.Round(10*time.Millisecond))
	}
	return msg
}

// Calc exponentional moving average
type EMA float64

func (e EMA) Add(val float64) float64 {
	if e == 0 {
		return val
	}
	return val*emaK + float64(e)*(1-emaK)
}

// Running mean and stddev as per http://www.johndcook.com/blog/standard_deviation/
type Average struct {
	Count, Mean float64
	Var, StdDev float64
	oldM, oldV  float64
}

func (s *Average) Add(x float64) {
	s.Count++
	if s.Count == 1 {
		s.oldM, s.Mean = x, x
		s.oldV = 0
	} else {
		s.Mean = s.oldM + (x-s.oldM)/s.Count
		s.Var = s.oldV + (x-s.oldM)*(x-s.Mean)
		s.oldM, s.oldV = s.Mean, s.Var
		if s.Count > 1 {
			s.StdDev = math.Sqrt(s.Var / (s.Count - 1))
		}
	}
}

func (s *Average) HTML() template.HTML {
	var text string
	if s.Mean > 10 {
		if s.StdDev < 0.1 {
			text = fmt.Sprintf("%.1f", s.Mean)
		} else {
			text = fmt.Sprintf("%.1f&PlusMinus;%.1f", s.Mean, s.StdDev)
		}
	} else {
		if s.StdDev < 0.01 {
			text = fmt.Sprintf("%.2f", s.Mean)
		} else {
			text = fmt.Sprintf("%.2f&PlusMinus;%.2f", s.Mean, s.StdDev)
		}
	}
	return template.HTML(text)
}

// Tester interface to evaluate the performance after each epoch, Test method returns true if training should stop.
type Tester interface {
	Test(net *Network, epoch int, loss float64, start time.Time) bool
	Epilogue() bool
}

// Tester which evaluates the loss and error for each of the data sets and updates the stats.
type TestBase struct {
	Net      *Network
	Data     map[string]*Dataset
	Pred     map[string][]int32
	Stats    []Stats
	Headers  []string
	Samples  int
	epilogue bool
}

// Create a new base class which implements the Tester interface.
func NewTestBase() *TestBase {
	return &TestBase{Stats: []Stats{}, Data: map[string]*Dataset{}}
}

// Release allocated buffers
func (t *TestBase) Release() {
	if t.Net != nil {
		t.Net.Release()
	}
	for _, dset := range t.Data {
		dset.Release()
	}
}

// Initialise the test dataset, network and other configuration.
func (t *TestBase) Init(queue num.Queue, conf Config, data map[string]Data, rng *rand.Rand) *TestBase {
	t.Data = make(map[string]*Dataset)
	t.Headers = StatsHeaders(data)
	t.Samples = min(conf.MaxSamples, data["train"].Len())
	t.Pred = nil
	t.epilogue = false
	if conf.DebugLevel >= 1 {
		fmt.Printf("init tester: samples=%d batch size=%d\n", t.Samples, conf.TestBatch)
	}
	for key, d := range data {
		if conf.DebugLevel >= 1 {
			fmt.Println("dataset =>", key)
		}
		t.Data[key] = NewDataset(queue.Dev(), d, conf.TestBatch, t.Samples, conf.FlattenInput, rng)
	}
	t.Net = New(queue, conf, t.Data["train"].BatchSize, t.Data["train"].Shape(), rng)
	return t
}

// Generate the predicted results when test is next run.
func (t *TestBase) Predict() *TestBase {
	t.Pred = make(map[string][]int32)
	for key, dset := range t.Data {
		t.Pred[key] = make([]int32, dset.Samples)
	}
	return t
}

// Reset stats prior to new run
func (t *TestBase) Reset() {
	t.Stats = t.Stats[:0]
	t.epilogue = false
}

// Test performance of the network, called from the Train function on completion of each epoch.
func (t *TestBase) Test(net *Network, epoch int, loss float64, start time.Time) bool {
	net.CopyTo(t.Net, false)
	if net.DebugLevel >= 1 {
		fmt.Printf("== TEST EPOCH %d ==\n", epoch)
	}
	s := Stats{Epoch: epoch, Values: []float64{loss}, BestSince: -1}
	for ix, key := range DataTypes {
		if dset, ok := t.Data[key]; ok {
			if dset.Samples < dset.Len() {
				dset.Shuffle()
			}
			var pred []int32
			if t.Predict != nil {
				pred = t.Pred[key]
			}
			errVal := t.Net.Error(dset, pred)
			s.Values = append(s.Values, errVal)
			if key == "valid" {
				// save average validation error
				avgVal := 0.0
				if epoch > 1 {
					avgVal = t.Stats[epoch-2].Values[ix+2]
				}
				avgVal = EMA(avgVal).Add(errVal)
				s.Values = append(s.Values, avgVal)
				// get number of epochs where average validation error has increased
				for ep := epoch - 1; ep >= 1; ep-- {
					prevErr := t.Stats[ep-1].Values[ix+2]
					if prevErr > avgVal {
						s.BestSince = epoch - ep - 1
						break
					}
				}
			}
		}
	}
	s.Elapsed = time.Since(start)
	if len(t.Stats) > 0 {
		s.Elapsed += t.Stats[len(t.Stats)-1].Elapsed
	}
	t.Stats = append(t.Stats, s)
	done := false
	if epoch >= net.MaxEpoch || loss <= net.MinLoss {
		done = true
	} else if net.StopAfter > 0 && s.BestSince >= net.StopAfter {
		// auto stopping based on performance on validation set
		if net.ExtraEpochs > 0 {
			// rewind data and perform additional training on undistorted training samples
			fmt.Printf("training for %d extra epochs\n", net.ExtraEpochs)
			t.epilogue = true
			net.StopAfter = 0
			net.MaxEpoch = epoch + net.ExtraEpochs - 1
		} else {
			done = true
		}
	}
	return done
}

func (t *TestBase) Epilogue() bool {
	return t.epilogue
}

type testLogger struct {
	*TestBase
}

// Create a new tester which logs stats to stdout.
func NewTestLogger(queue num.Queue, conf Config, data map[string]Data, rng *rand.Rand) Tester {
	return testLogger{TestBase: NewTestBase().Init(queue, conf, data, rng)}
}

func (t testLogger) Test(net *Network, epoch int, loss float64, start time.Time) bool {
	done := t.TestBase.Test(net, epoch, loss, start)
	s := t.Stats[len(t.Stats)-1]
	if done || net.LogEvery == 0 || epoch%net.LogEvery == 0 {
		fmt.Println(s.String(t.Headers, done))
	}
	return done
}

// Train the network on the given training set by updating the weights
func Train(net *Network, dset *Dataset, test Tester) {
	acc := net.queue.NewArray(num.Float32)
	done := false
	for epoch := 1; epoch <= net.MaxEpoch && !done; epoch++ {
		if test.Epilogue() {
			dset.Rewind()
		}
		start := time.Now()
		loss := TrainEpoch(net, dset, acc)
		done = test.Test(net, epoch, loss, start)
	}
}

// Perform one training epoch on dataset, returns the current loss prior to updating the weights.
func TrainEpoch(net *Network, dset *Dataset, acc num.Array) float64 {
	q := net.queue
	if net.inputGrad == nil {
		net.inputGrad = q.NewArray(num.Float32, len(dset.Classes()), dset.BatchSize)
	}
	if net.Shuffle {
		dset.Shuffle()
	}
	weightDecay := float32(net.Eta*net.Lambda) / float32(dset.Samples)
	q.Call(num.Fill(acc, 0))
	dset.NextEpoch()
	for batch := 0; batch < dset.Batches; batch++ {
		if net.DebugLevel >= 2 || (net.DebugLevel == 1 && batch == 0) {
			fmt.Printf("== train batch %d ==\n", batch)
		}
		q.Finish()
		x, _, yOneHot := dset.NextBatch()
		yPred := net.Fprop(x, true)
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
