package nnet

import (
	"fmt"
	"github.com/jnb666/deepthought2/num"
	"github.com/jnb666/deepthought2/stats"
	"log"
	"math/rand"
	"time"
)

// Training statistics
type Stats struct {
	Epoch     int
	AvgLoss   float64
	Loss      []float64
	Error     []float64
	BestSince int
	TrainTime time.Duration
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
	stats.Loss = append([]float64{}, s.Loss...)
	stats.Error = append([]float64{}, s.Error...)
	return stats
}

func (s Stats) Format() []string {
	str := []string{s.FormatLoss()}
	for i := range s.Error {
		str = append(str, s.FormatError(i))
	}
	return str
}

func (s Stats) FormatLoss() string {
	return fmt.Sprintf("%7.4f", s.AvgLoss)
}

func (s Stats) FormatError(i int) string {
	if s.Error[i] >= 0.1 {
		return fmt.Sprintf("%6.1f%%", s.Error[i]*100)
	}
	return fmt.Sprintf("%6.2f%%", s.Error[i]*100)
}

func (s Stats) FormatElapsed() string {
	return FormatDuration(s.Elapsed)
}

func (s Stats) String(headers []string) string {
	msg := fmt.Sprintf("epoch %3d:", s.Epoch)
	for i, val := range s.Format() {
		msg += fmt.Sprintf("  %s =%s", headers[i], val)
	}
	return msg
}

func FormatDuration(d time.Duration) string {
	if d >= time.Minute {
		return d.Round(time.Second).String()
	}
	return fmt.Sprintf("%.2fs", d.Seconds())
}

// Tester interface to evaluate the performance after each epoch, Test method returns true if training should stop.
type Tester interface {
	Test(net *Network, epoch int, batchLoss []float64, trainError float64, start time.Time) bool
	Epilogue() bool
	Memory() int
	MemoryProfile() string
	Release()
}

// Tester which evaluates the loss and error for each of the data sets and updates the stats.
type TestBase struct {
	Net      *Network
	Data     map[string]*Dataset
	Pred     map[string][]int32
	Stats    []Stats
	Headers  []string
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
		t.Net = nil
	}
	for _, dset := range t.Data {
		dset.Release()
	}
	t.Data = nil
}

func (t *TestBase) Memory() int {
	if t.Net == nil {
		return 0
	}
	return t.Net.Memory()
}

func (t *TestBase) MemoryProfile() string {
	if t.Net == nil {
		return ""
	}
	return t.Net.MemoryProfile()
}

// Initialise the test dataset, network and other configuration.
func (t *TestBase) Init(dev num.Device, conf Config, data map[string]Data, rng *rand.Rand) *TestBase {
	opts := conf.DatasetConfig(true)
	t.Data = make(map[string]*Dataset)
	t.Headers = StatsHeaders(data)
	t.Pred = nil
	t.epilogue = false
	if debug >= 1 {
		log.Printf("init tester: samples=%d batch size=%d\n", opts.MaxSamples, opts.BatchSize)
	}
	for key, d := range data {
		if key != "train" {
			if debug >= 1 {
				log.Println("dataset =>", key)
			}
			t.Data[key] = NewDataset(dev, d, opts, rng)
			t.Data[key].Profiling(conf.Profile, "tester:"+key)
		}
	}
	if opts.BatchSize != conf.TrainBatch {
		t.Net = New(dev.NewQueue(), conf, t.Data["test"].BatchSize, t.Data["test"].Shape(), false, rng)
		log.Println("allocate test network: input shape ", t.Net.InShape)
	}
	return t
}

// Generate the predicted results when test is next run.
func (t *TestBase) Predict(train *Dataset) *TestBase {
	t.Pred = make(map[string][]int32)
	for key, dset := range t.Data {
		t.Pred[key] = make([]int32, dset.Samples)
	}
	t.Pred["train"] = make([]int32, train.Samples)
	return t
}

// Reset stats prior to new run
func (t *TestBase) Reset() {
	t.Stats = t.Stats[:0]
	t.epilogue = false
}

// Test performance of the network, called from the Train function on completion of each epoch.
func (t *TestBase) Test(net *Network, epoch int, batchLoss []float64, trainError float64, start time.Time) bool {
	s := Stats{
		Epoch:     epoch,
		Loss:      append([]float64{}, batchLoss...),
		Error:     []float64{trainError},
		BestSince: -1,
		TrainTime: time.Since(start),
	}
	for _, loss := range batchLoss {
		s.AvgLoss += loss
	}
	s.AvgLoss /= float64(len(batchLoss))
	if t.Net != nil {
		// copy the weights to net with different input shape
		CopyParams(net.queue, net.Layers, t.Net.Layers, true)
		net = t.Net
	}
	if debug >= 1 {
		log.Printf("== TEST EPOCH %d ==\n", epoch)
	}
	for ix, key := range DataTypes {
		if dset, ok := t.Data[key]; ok {
			if dset.Samples < dset.Len() {
				dset.Shuffle()
			}
			var pred []int32
			if t.Predict != nil {
				pred = t.Pred[key]
			}
			errVal := net.Error(dset, pred)
			s.Error = append(s.Error, errVal)
			if key == "valid" {
				// save average validation error
				avgVal := 0.0
				if epoch > 1 {
					avgVal = t.Stats[epoch-2].Error[ix+1]
				}
				avgVal = stats.EMA(avgVal).Add(errVal, net.ValidEMA)
				s.Error = append(s.Error, avgVal)
				// get number of epochs where average validation error has increased
				for ep := epoch - 1; ep >= 1; ep-- {
					prevErr := t.Stats[ep-1].Error[ix+1]
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
		s.TrainTime += t.Stats[len(t.Stats)-1].TrainTime
		s.Elapsed += t.Stats[len(t.Stats)-1].Elapsed
	}
	t.Stats = append(t.Stats, s)
	done := false
	loss := batchLoss[len(batchLoss)-1]
	if epoch >= net.MaxEpoch || loss <= net.MinLoss || (net.MaxSeconds > 0 && int(s.Elapsed.Seconds()) > net.MaxSeconds) {
		done = true
	} else if net.StopAfter > 0 && s.BestSince >= net.StopAfter && epoch > 10 {
		// auto stopping based on performance on validation set
		if net.ExtraEpochs > 0 {
			// perform additional training on undistorted training samples
			t.epilogue = true
			net.StopAfter = 0
			net.MaxEpoch = epoch + net.ExtraEpochs
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
func NewTestLogger(dev num.Device, conf Config, data map[string]Data, rng *rand.Rand) Tester {
	return testLogger{TestBase: NewTestBase().Init(dev, conf, data, rng)}
}

func (t testLogger) Test(net *Network, epoch int, batchLoss []float64, trainErr float64, start time.Time) bool {
	done := t.TestBase.Test(net, epoch, batchLoss, trainErr, start)
	s := t.Stats[len(t.Stats)-1]
	if done || net.LogEvery == 0 || epoch%net.LogEvery == 0 {
		log.Println(s.String(t.Headers))
		if done {
			log.Printf("train time:%s  total:%s", FormatDuration(s.TrainTime), FormatDuration(s.Elapsed))
		}
	}
	return done
}

// Train the network on the given training set by updating the weights
func Train(net *Network, dset *Dataset, test Tester) {
	done := false
	epilogue := false
	for epoch := 1; epoch <= net.MaxEpoch && !done; epoch++ {
		if test.Epilogue() && !epilogue {
			log.Printf("training for %d extra epochs\n", net.ExtraEpochs)
			dset.SetTrans(net.Normalise, false)
			epilogue = true
		}
		start := time.Now()
		batchLoss, trainError := TrainEpoch(net, dset, epoch, nil)
		done = test.Test(net, epoch, batchLoss, trainError, start)
	}
	if epilogue {
		dset.SetTrans(net.Normalise, net.Distort)
	}
}

// Perform one training epoch on dataset, returns the current loss prior to updating the weights.
func TrainEpoch(net *Network, dset *Dataset, epoch int, pred []int32) (batchLoss []float64, trainError float64) {
	q := net.queue
	if net.Shuffle {
		dset.Shuffle()
	}
	learningRate, weightDecay := net.OptimiserParams(epoch, dset.Samples)
	optimiser := SGD{
		LearningRate: float32(learningRate / float64(dset.BatchSize)),
		WeightDecay:  float32(weightDecay),
		Momentum:     float32(net.Momentum),
		Nesterov:     net.Nesterov,
	}
	var p []int32
	if pred != nil {
		p = make([]int32, dset.Samples)
	}
	dset.NextEpoch()
	// if batch size < 64 then average loss over 10 batches
	lossBatches := 1
	nloss := dset.Batches
	if dset.BatchSize < 64 {
		lossBatches = 10
		nloss /= lossBatches
		if dset.Batches%lossBatches != 0 {
			nloss++
		}
	}
	batchLoss = make([]float64, nloss)
	for batch := 0; batch < dset.Batches; batch++ {
		if debug >= 2 || (debug == 1 && batch == 0) {
			log.Printf("== train batch %d ==\n", batch)
		}
		q.Finish()
		x, y, yOneHot := dset.NextBatch()
		// forward propagation
		yPred := Fprop(q, net.Layers, x, net.WorkSpace[0], true)
		if debug >= 2 {
			log.Printf("yOneHot:\n%s", yOneHot.String(q))
			log.Printf("yPred\n%s", yPred.String(q))
		}
		// sum average loss and error over batches
		batchLoss[batch/lossBatches] += net.BatchLoss(yOneHot, yPred) / float64(lossBatches)
		trainError += net.BatchError(batch, dset, y, yPred, p)
		// get difference at output and  back propagate gradient and update weights
		grad := num.NewArray(net.WorkSpace[1], num.Float32, yOneHot.Dims...)
		q.Call(
			num.Copy(yPred, grad),
			num.Axpy(-1, yOneHot, grad),
		)
		if debug >= 2 {
			log.Printf("input grad:\n%s", grad.String(q))
		}
		Bprop(q, net.Layers, grad, net.WorkSpace)
		ParamLayers("", net.Layers, func(desc string, l ParamLayer) {
			l.UpdateParams(q, optimiser, net.WorkSpace[0])
		})
		if debug >= 2 || (batch == dset.Batches-1 && debug >= 1) {
			net.PrintWeights()
		}
	}
	if pred != nil {
		for i, ix := range dset.indexes {
			pred[ix] = p[i]
		}
	}
	return batchLoss, trainError / float64(dset.Samples)
}

// Optimiser updates the weights
type Optimiser interface {
	Update(q num.Queue, decay bool, x, dx, v *num.Array, work num.Buffer)
}

// Vanilla stockastic gradient descent
type SGD struct {
	LearningRate float32
	WeightDecay  float32
	Momentum     float32
	Nesterov     bool
}

func (o SGD) Update(q num.Queue, decay bool, x, dx, v *num.Array, work num.Buffer) {
	q.Call(num.Scale(-o.LearningRate, dx))
	if decay && o.WeightDecay != 0 {
		q.Call(num.Axpy(-o.WeightDecay, x, dx))
	}
	switch {
	case o.Momentum != 0 && o.Nesterov:
		// v = dx + mom*v; x += -mom*vPrev + (1 + mom)*v
		vPrev := num.NewArray(work, num.Float32, v.Dims...)
		q.Call(
			num.Copy(v, vPrev),
			num.Scale(o.Momentum, v),
			num.Axpy(1, dx, v),
			num.Axpy(1+o.Momentum, v, x),
			num.Axpy(-o.Momentum, vPrev, x),
		)
	case o.Momentum != 0 && !o.Nesterov:
		// v = mom*v + dx; x += v
		q.Call(
			num.Scale(o.Momentum, v),
			num.Axpy(1, dx, v),
			num.Axpy(1, v, x),
		)
	default:
		q.Call(num.Axpy(1, dx, x))
	}
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
