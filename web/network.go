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

// Network configuration
type Config struct {
	nnet.Config
	Model string
}

// Load configuration data
func NewConfig(model string) (*Config, error) {
	log.Println("load model:", model)
	conf, err := nnet.LoadConfig(model + ".net")
	return &Config{Config: conf, Model: model}, err
}

// Network and associated training / test data and configuration
type Network struct {
	*nnet.Network
	*Tester
	Data      map[string]nnet.Data
	Conf      *Config
	trainData *nnet.Dataset
	queue     num.Queue
}

// Load config and data given model name
func NewNetwork(conf *Config) (*Network, error) {
	var err error
	n := &Network{Conf: conf}
	if n.Data, err = nnet.LoadData(conf.DataSet); err != nil {
		return nil, err
	}
	dev := num.NewDevice(conf.UseGPU)
	n.queue = dev.NewQueue()
	n.Network = nnet.New(n.queue, n.Conf.Config, n.Conf.TrainBatch, n.Data["train"].Shape)
	if n.DebugLevel >= 1 {
		fmt.Println(n.Network)
	}
	n.InitWeights()
	n.Tester = NewTester(n.queue, n.Conf.Config, n.Data)
	return n, nil
}

// Perform training run
func (n *Network) Train(restart bool) {
	log.Printf("train: start %s - restart=%v\n", n.Conf.Model, restart)
	if restart {
		n.Network = nnet.New(n.queue, n.Conf.Config, n.Conf.TrainBatch, n.Data["train"].Shape)
		n.Tester.Init(n.queue, n.Data, n.Conf.Config)
		n.InitWeights()
		n.trainData = nnet.NewDataset(n.queue.Dev(), n.Data["train"], n.TrainBatch, n.MaxSamples)
		n.Epoch = 1
	} else if n.Epoch > 0 {
		n.Epoch++
	}
	if n.Epoch == 0 || n.Epoch > n.MaxEpoch {
		return
	}
	n.running = true
	go func() {
		n.queue.Profiling(n.Profile)
		acc := n.queue.NewArray(num.Float32)
		epoch := n.Epoch
		done := false
		start := time.Now()
		for !done {
			loss := nnet.TrainEpoch(n.Network, n.trainData, acc)
			done = n.Tester.Test(n.Network, epoch, loss, start)
			epoch++
		}
		n.Lock()
		n.running = false
		n.Unlock()
		log.Println("train: end")
		if n.Profile {
			fmt.Printf("== Profile ==\n%s\n", n.queue.Profile())
		}
	}()
}

// Network tester which evaluates the error and stores the stats
type Tester struct {
	Epoch   int
	Stats   []nnet.Stats
	Pred    map[string][]int32
	base    *nnet.TestBase
	conn    *websocket.Conn
	running bool
	sync.Mutex
}

// Create new tester instance and get initial prediction
func NewTester(queue num.Queue, conf nnet.Config, data map[string]nnet.Data) *Tester {
	t := &Tester{Stats: []nnet.Stats{}, Pred: map[string][]int32{}}
	return t.Init(queue, data, conf)
}

// Initialise tester at start of run
func (t *Tester) Init(queue num.Queue, data map[string]nnet.Data, conf nnet.Config) *Tester {
	t.base = nnet.NewTestBase(queue, conf, data, false)
	t.base.Predict = map[string][]int32{}
	t.Epoch = 0
	t.Stats = t.Stats[:0]
	for key, dset := range t.base.Data {
		t.base.Predict[key] = make([]int32, dset.Samples)
		t.Pred[key] = make([]int32, dset.Samples)
		t.base.Net.Error(dset, t.Pred[key])
	}
	return t
}

// Update network stats and trigger websocket message to refresh the page
func (t *Tester) Test(net *nnet.Network, epoch int, loss float64, start time.Time) bool {
	done := t.base.Test(net, epoch, loss, start)
	t.Lock()
	t.Epoch = epoch
	t.Stats = append(t.Stats, t.base.Stats)
	for key, pred := range t.base.Predict {
		copy(t.Pred[key], pred)
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
