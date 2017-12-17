// Package web has a web based interface we network training and visualisation.
package web

import (
	"encoding/gob"
	"fmt"
	"github.com/gorilla/websocket"
	"github.com/jnb666/deepthought2/img"
	"github.com/jnb666/deepthought2/nnet"
	"github.com/jnb666/deepthought2/num"
	"html/template"
	"log"
	"math/rand"
	"os"
	"path"
	"strconv"
	"sync"
	"time"
)

// Network and associated training / test data and configuration
type Network struct {
	*NetworkData
	*nnet.Network
	*Tester
	Data      map[string]nnet.Data
	Labels    map[string][]int32
	trans     *img.Transformer
	conn      *websocket.Conn
	trainData *nnet.Dataset
	queue     num.Queue
	rng       *rand.Rand
}

// Embedded structs used to persist state to file
type NetworkData struct {
	Model  string
	Conf   nnet.Config
	Epoch  int
	Stats  []nnet.Stats
	Pred   map[string][]int32
	Params []LayerData
}

type LayerData struct {
	Layer   int
	Weights []float32
	Biases  []float32
}

// Create a new network and load config from data given model name
func NewNetwork(model string) (*Network, error) {
	n := &Network{Tester: NewTester()}
	log.Println("load model:", model)
	var err error
	n.NetworkData, err = LoadNetwork(model, false)
	if err != nil {
		return nil, err
	}
	if err := n.Init(); err != nil {
		return nil, err
	}
	if err := n.Import(); err != nil {
		return nil, err
	}
	return n, nil
}

// Initialise the network
func (n *Network) Init() error {
	log.Printf("init network: dataSet=%s useGPU=%v\n", n.Conf.DataSet, n.Conf.UseGPU)
	var err error
	if n.Data, err = nnet.LoadData(n.Conf.DataSet); err != nil {
		return err
	}
	dev := num.NewDevice(n.Conf.UseGPU)
	n.queue = dev.NewQueue()
	n.rng = nnet.SetSeed(n.Conf.RandSeed)
	n.trainData = nnet.NewDataset(n.queue.Dev(), n.Data["train"], n.Conf.TrainBatch, n.Conf.MaxSamples, n.Conf.FlattenInput, n.rng)
	n.Network = nnet.New(n.queue, n.Conf, n.trainData.BatchSize, n.trainData.Shape())
	if n.DebugLevel >= 1 {
		fmt.Println(n.Network)
	}
	n.Labels = n.initTester(n.queue, n.Data, n.Conf)
	dims := n.trainData.Shape()
	if len(dims) >= 2 {
		n.trans = img.NewTransformer(dims[1], dims[0], img.ConvAccel, n.testRng)
	}
	return nil
}

// Perform training run
func (n *Network) Train(restart bool) error {
	log.Printf("train: start %s - restart=%v\n", n.Model, restart)
	if restart {
		if err := n.Init(); err != nil {
			return err
		}
		n.InitWeights(n.rng)
		n.Tester.base.Reset()
		n.Epoch = 1
	} else if n.Epoch > 0 {
		n.Epoch++
	}
	if n.Epoch == 0 || n.Epoch > n.MaxEpoch {
		return nil
	}
	n.running = true
	n.stop = false
	go func() {
		n.queue.Profiling(n.Profile)
		acc := n.queue.NewArray(num.Float32)
		done := false
		epoch := n.Epoch
		start := time.Now()
		for !done {
			loss := nnet.TrainEpoch(n.Network, n.trainData, acc)
			done = n.Tester.Test(n.Network, epoch, loss, start)
			epoch = n.nextEpoch(epoch)
		}
		n.Lock()
		n.running = false
		n.stop = false
		n.Unlock()
		log.Println("train: end")
		if n.Profile {
			fmt.Printf("== Profile ==\n%s\n", n.queue.Profile())
		}
	}()
	return nil
}

func (n *Network) nextEpoch(epoch int) int {
	n.Lock()
	n.Epoch = epoch
	n.predict(n.Pred)
	n.Unlock()
	// notify via websocket
	if n.conn != nil {
		msg := []byte(strconv.Itoa(epoch))
		err := n.conn.WriteMessage(websocket.TextMessage, msg)
		if err != nil {
			log.Println("nextEpoch: error writing to websocket", err)
		}
	} else {
		log.Println("nextEpoch: websocket is not initialised")
	}
	// save state to disk
	n.Lock()
	n.Export()
	err := SaveNetwork(n.NetworkData)
	n.Unlock()
	if err != nil {
		log.Println("nextEpoch: error saving network:", err)
	}
	return epoch + 1
}

func (n *Network) heading() template.HTML {
	s := fmt.Sprintf(`%s: epoch <span id="epoch">%d</span> of %d`, n.Model, n.Epoch, n.MaxEpoch)
	return template.HTML(s)
}

// Export current state prior to saving to file
func (n *Network) Export() {
	n.Stats = n.base.Stats
	n.Params = []LayerData{}
	if n.base.Net == nil || n.base.Net.Layers == nil {
		return
	}
	for i, layer := range n.base.Net.Layers {
		if l, ok := layer.(nnet.ParamLayer); ok {
			W, B := l.Params()
			d := LayerData{
				Layer:   i,
				Weights: make([]float32, num.Prod(W.Dims())),
				Biases:  make([]float32, num.Prod(B.Dims())),
			}
			n.queue.Call(
				num.Read(W, d.Weights),
				num.Read(B, d.Biases),
			)
			n.Params = append(n.Params, d)
		}
	}
	if len(n.Params) > 0 {
		n.queue.Finish()
	}
}

// Import current state after loading from file
func (n *Network) Import() error {
	n.base.Stats = n.Stats
	if n.Epoch == 0 {
		log.Println("init weights")
		n.InitWeights(n.rng)
	} else if n.Params != nil && len(n.Params) > 0 {
		log.Println("import weights")
		nlayers := len(n.Network.Layers)
		for _, p := range n.Params {
			if p.Layer >= nlayers {
				return fmt.Errorf("layer %d import error: network has %d layers total", p.Layer, nlayers)
			}
			layer, ok := n.Network.Layers[p.Layer].(nnet.ParamLayer)
			if !ok {
				return fmt.Errorf("layer %d import error: not a ParamLayer", p.Layer)
			}
			W, B := layer.Params()
			wsize, bsize := num.Prod(W.Dims()), num.Prod(B.Dims())
			if wsize != len(p.Weights) || bsize != len(p.Biases) {
				return fmt.Errorf("layer %d import error: size mismatch - have %d %d - expect %d %d",
					p.Layer, len(p.Weights), len(p.Biases), wsize, bsize)
			}
			n.queue.Call(
				num.Write(W, p.Weights),
				num.Write(B, p.Biases),
			)
		}
		n.queue.Finish()
	}
	return nil
}

// Network tester which evaluates the error and stores the stats
type Tester struct {
	base    *nnet.TestBase
	testRng *rand.Rand
	running bool
	stop    bool
	sync.Mutex
}

// Create new tester instance and get initial prediction
func NewTester() *Tester {
	return &Tester{base: nnet.NewTestBase()}
}

// Initialise tester at start of run
func (t *Tester) initTester(queue num.Queue, data map[string]nnet.Data, conf nnet.Config) map[string][]int32 {
	t.testRng = nnet.SetSeed(conf.RandSeed)
	t.base.Init(queue, conf, data, t.testRng).Predict()
	labels := make(map[string][]int32)
	for key, dset := range t.base.Data {
		labels[key] = make([]int32, dset.Samples)
		dset.Label(seq(dset.Samples), labels[key])
	}
	return labels
}

// Update network stats and trigger websocket message to refresh the page
func (t *Tester) Test(net *nnet.Network, epoch int, loss float64, start time.Time) bool {
	done := t.base.Test(net, epoch, loss, start)
	t.Lock()
	if t.stop {
		t.stop = false
		t.running = false
		done = true
	}
	t.Unlock()
	return done
}

// Stop training run if in progress
func (t *Tester) Stop() {
	if t.running {
		t.stop = true
		for t.running {
			time.Sleep(50 * time.Millisecond)
		}
	}
}

// Export predicted values
func (t *Tester) predict(out map[string][]int32) {
	for key, pred := range t.base.Pred {
		if arr, ok := out[key]; !ok || len(arr) != len(pred) {
			out[key] = make([]int32, len(pred))
		}
		copy(out[key], pred)
	}
}

// Encode data in gob format and save to file under nnet.DataDir
func SaveNetwork(data *NetworkData) error {
	name := data.Model + ".net"
	filePath := path.Join(nnet.DataDir, name)
	f, err := os.Create(filePath)
	if err != nil {
		return err
	}
	defer f.Close()
	return gob.NewEncoder(f).Encode(*data)
}

// Read back gob encoded data file, if not found or reset is set then load default config.
func LoadNetwork(model string, reset bool) (data *NetworkData, err error) {
	data = &NetworkData{
		Model:  model,
		Stats:  []nnet.Stats{},
		Pred:   map[string][]int32{},
		Params: []LayerData{},
	}
	if !reset {
		if err = loadGob(model+".net", data); err != nil {
			reset = true
		}
	}
	if reset {
		data.Conf, err = nnet.LoadConfig(model + ".conf")
	}
	return data, err
}

func loadGob(name string, data *NetworkData) error {
	filePath := path.Join(nnet.DataDir, name)
	f, err := os.Open(filePath)
	if err != nil {
		return err
	}
	defer f.Close()
	log.Println("loading network config from", name)
	return gob.NewDecoder(f).Decode(data)
}
