// Package web has a web based interface we network training and visualisation.
package web

import (
	"fmt"
	"github.com/gorilla/websocket"
	"github.com/jnb666/deepthought2/nnet"
	"github.com/jnb666/deepthought2/num"
	"log"
	"strconv"
	"sync"
	"time"
)

// Network and associated training / test data and configuration
type Network struct {
	*nnet.Network
	*Tester
	Data      map[string]nnet.Data
	trainData *nnet.Dataset
	queue     num.Queue
}

// Load config and data given model name
func NewNetwork(dev num.Device, model string) (*Network, error) {
	n := &Network{}
	fmt.Println("load model:", model)
	conf, err := nnet.LoadConfig(model)
	if err != nil {
		return nil, err
	}
	n.Data, err = nnet.LoadData(model)
	if err != nil {
		return nil, err
	}
	n.queue = dev.NewQueue(conf.Threads)
	n.Network = nnet.New(dev, conf, conf.TrainBatch, n.Data["train"].Shape)
	n.InitWeights(n.queue)
	n.Tester = NewTester(n.queue, conf, n.Data)
	return n, nil
}

// Perform training run
func (n *Network) Train(conf nnet.Config, restart bool) {
	log.Println("train: start - restart =", restart)
	if restart {
		nnet.SetSeed(conf.RandSeed)
		n.Network = nnet.New(n.queue.Dev(), conf, conf.TrainBatch, n.Data["train"].Shape)
		n.Tester.Init(n.queue.Dev(), n.Data, conf)
		n.InitWeights(n.queue)
		n.trainData = nnet.NewDataset(n.queue.Dev(), n.Data["train"], conf.TrainBatch, conf.MaxSamples)
		n.Epoch = 1
	} else if n.Epoch > 0 {
		n.MaxEpoch = conf.MaxEpoch
		n.MinLoss = conf.MinLoss
		n.Epoch++
	}
	if n.Epoch == 0 || n.Epoch >= n.MaxEpoch {
		return
	}
	n.running = true
	go func() {
		acc := n.queue.NewArray(num.Float32)
		epoch := n.Epoch
		done := false
		start := time.Now()
		for !done {
			loss := nnet.TrainEpoch(n.queue, n.Network, n.trainData, acc)
			done = n.Tester.Test(n.queue, n.Network, epoch, loss, start)
			epoch++
		}
		n.Lock()
		n.running = false
		n.Unlock()
		log.Println("train: end")
	}()
}

// Network tester which evaluates the error and stores the stats
type Tester struct {
	Stats   []nnet.Stats
	Epoch   int
	net     *nnet.Network
	data    []*nnet.Dataset
	pred    [][]int32
	conn    *websocket.Conn
	running bool
	sync.Mutex
}

// Create new tester instance and get initial prediction
func NewTester(q num.Queue, conf nnet.Config, data map[string]nnet.Data) *Tester {
	t := &Tester{Stats: []nnet.Stats{}, data: []*nnet.Dataset{}, pred: [][]int32{}}
	t.Init(q.Dev(), data, conf)
	for i, dset := range t.data {
		t.net.Error(q, dset, t.pred[i])
	}
	return t
}

// Initialise tester at start of run
func (t *Tester) Init(dev num.Device, data map[string]nnet.Data, conf nnet.Config) {
	t.net = nnet.New(dev, conf, conf.TestBatch, data["train"].Shape)
	t.Epoch = 0
	t.Stats = t.Stats[:0]
	t.data = t.data[:0]
	t.pred = t.pred[:0]
	for _, key := range nnet.DataTypes {
		if d, ok := data[key]; ok {
			dset := nnet.NewDataset(dev, d, conf.TestBatch, conf.MaxSamples)
			t.data = append(t.data, dset)
			t.pred = append(t.pred, make([]int32, dset.Samples))
		}
	}
}

// Update network stats and trigger websocket message to refresh the page
func (t *Tester) Test(q num.Queue, net *nnet.Network, epoch int, loss float64, start time.Time) bool {
	net.CopyTo(q, t.net)
	var s nnet.Stats
	s.Epoch = epoch
	s.Values = []float64{loss}
	pred := [][]int32{}
	for i, dset := range t.data {
		pred = append(pred, make([]int32, dset.Samples))
		s.Values = append(s.Values, t.net.Error(q, dset, pred[i]))
	}
	s.Elapsed = time.Since(start)
	done := epoch >= net.MaxEpoch || loss <= net.MinLoss
	t.Lock()
	t.Epoch = epoch
	t.Stats = append(t.Stats, s)
	for i := range pred {
		copy(t.pred[i], pred[i])
	}
	if !t.running {
		done = true
	}
	t.Unlock()
	if t.conn != nil {
		msg := []byte(strconv.Itoa(epoch))
		err := t.conn.WriteMessage(websocket.TextMessage, msg)
		if err != nil {
			log.Println("Test: error writing to websocket", err)
		}
	} else {
		log.Println("Test: websocket is not initialised")
	}
	return done
}

// Get predicted output for each image
func (t *Tester) Pred(dset string) []int32 {
	for i, typ := range nnet.DataTypes {
		if dset == typ && i < len(t.pred) {
			return t.pred[i]
		}
	}
	return nil
}
