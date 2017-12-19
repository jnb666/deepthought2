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
	"image"
	"image/color"
	"log"
	"math"
	"math/rand"
	"os"
	"path"
	"strconv"
	"sync"
	"time"
)

const (
	aspectOutput     = 0.125
	aspectWeights    = 0.25
	factorMinOutput  = 20
	factorMinWeights = 20
)

// color map definition
const cmin = -1
const cmax = 1

var cmap = [][3]float32{{0, 0, .5}, {0, 0, 1}, {0, .5, 1}, {0, 1, 1}, {.5, 1, .5}, {1, 1, 0}, {1, .5, 0}, {1, 0, 0}, {.5, 0, 0}}

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
	view      *viewData
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
	n.view = newViewData(n.queue, n.Data, n.Conf)
	return nil
}

// Initialise for new training run
func (n *Network) Start() error {
	if err := n.Init(); err != nil {
		return err
	}
	n.Tester.base.Reset()
	n.InitWeights(n.rng)
	n.Network.CopyTo(n.view.Network, true)
	n.Epoch = 0
	return nil
}

// Perform training run
func (n *Network) Train(restart bool) error {
	log.Printf("train: start %s - restart=%v\n", n.Model, restart)
	if restart {
		if err := n.Start(); err != nil {
			return err
		}
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
	n.Network.CopyTo(n.view.Network, true)
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
		n.Network.CopyTo(n.view.Network, true)
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

// Data used for network visualtion of weights and outputs
type viewData struct {
	*nnet.Network
	queue   num.Queue
	layers  []viewLayer
	dset    string
	data    nnet.Data
	input   num.Array
	inShape []int
	inData  []float32
	inImage *image.NRGBA
}

type viewLayer struct {
	ltype    string
	outShape []int
	outData  []float32
	outImage *image.NRGBA
	ox, oy   int
	wShape   []int
	bShape   []int
	wData    []float32
	bData    []float32
	wImage   *image.NRGBA
	wix, wiy int
	wox, woy int
	wborder  int
}

func newViewData(dev num.Device, data map[string]nnet.Data, conf nnet.Config) *viewData {
	v := &viewData{queue: dev.NewQueue()}
	if _, ok := data["test"]; ok {
		v.dset, v.data = "test", data["test"]
	} else {
		v.dset, v.data = "train", data["train"]
	}
	v.Network = nnet.New(v.queue, conf, 1, v.data.Shape())

	v.inShape = v.data.Shape()
	v.inData = make([]float32, num.Prod(v.inShape))
	if conf.FlattenInput {
		v.input = dev.NewArray(num.Float32, len(v.inData), 1)
	} else {
		v.input = dev.NewArray(num.Float32, append(v.inShape, 1)...)
	}
	v.inShape = v.inShape[:len(v.inShape)-1]
	if len(v.inShape) == 2 {
		v.inImage = image.NewNRGBA(image.Rect(0, 0, v.inShape[1], v.inShape[0]))
	}

	for i, layer := range v.Layers {
		l := viewLayer{ltype: layer.Type()}
		// filter output layers
		if l.ltype != "maxPool" {
			l.outShape = layer.OutShape()
			l.outShape = l.outShape[:len(l.outShape)-1]
		}
		prev := len(v.layers) - 1
		if prev >= 0 && layer.IsActiv() && num.SameShape(v.layers[prev].outShape, l.outShape) {
			l.ltype = v.layers[prev].ltype + " " + l.ltype
			v.layers[prev].outShape = nil
		}
		// allocate buffers and images for weights and biases
		if pLayer, ok := layer.(nnet.ParamLayer); ok {
			W, B := pLayer.Params()
			l.addWeightImage(i, W.Dims(), B.Dims())
		}
		v.layers = append(v.layers, l)
	}
	// allocate buffers and output images
	for i, l := range v.layers {
		if l.outShape != nil {
			v.layers[i].addOutputImage(i, l.outShape)
		}
	}
	return v
}

// update outputs with given index from test set and update the images
func (v *viewData) update(index int) {
	v.data.Input([]int{index}, v.inData)
	v.queue.Call(
		num.Write(v.input, v.inData),
	)
	v.Fprop(v.input)

	for i, l := range v.layers {
		if l.outImage != nil {
			v.queue.Call(
				num.Read(v.Layers[i].Output(), l.outData),
			).Finish()
			// draw output
			switch len(l.outShape) {
			case 1:
				height := l.outImage.Bounds().Dy()
				for i, val := range l.outData {
					l.outImage.Set(i/height, i%height, mapColor(val))
				}
			case 3:
				bw, bh := l.outShape[0], l.outShape[1]
				for i := 0; i < l.ox*l.oy; i++ {
					xb := (bw + 1) * (i % l.ox)
					yb := (bh + 1) * (i / l.ox)
					for j := 0; j < bw*bh; j++ {
						col := mapColor(l.outData[i*bw*bh+j])
						l.outImage.Set(xb+j%bw+1, yb+j/bw+1, col)
					}
				}
			}
		}
		if l.wImage != nil {
			W, B := v.Layers[i].(nnet.ParamLayer).Params()
			v.queue.Call(
				num.Read(W, l.wData),
				num.Read(B, l.bData),
			).Finish()
			// draw bias
			for i := 0; i < l.wox*l.woy; i++ {
				xb, yb := l.block(i)
				biasCol := mapColor(l.bData[i])
				for j := 0; j < l.wix; j++ {
					l.wImage.Set(xb+j, yb, biasCol)
				}
				for j := 0; j < l.wiy; j++ {
					l.wImage.Set(xb, yb+j, biasCol)
				}
			}
			// draw weights
			bsize := l.wix * l.wiy
			for i := 0; i < l.wox*l.woy; i++ {
				xb, yb := l.block(i)
				for j := 0; j < bsize; j++ {
					l.wImage.Set(xb+j%l.wix+1, yb+j/l.wix+1, mapColor(l.wData[i*bsize+j]))
				}
			}
		}
	}
}

func (v *viewData) lastLayer() *viewLayer {
	if len(v.layers) == 0 {
		return nil
	}
	return &v.layers[len(v.layers)-1]
}

func (l *viewLayer) addOutputImage(layer int, dims []int) {
	var width, height int
	switch len(dims) {
	case 1:
		// fully connected layer
		height, width = factorise(dims[0], factorMinOutput, aspectOutput)
	case 3:
		// convolutional layer
		l.oy, l.ox = factorise(dims[2], factorMinOutput, aspectOutput)
		height = (dims[1] + 1) * l.oy
		width = (dims[0] + 1) * l.ox
	default:
		log.Printf("viewLayer: output shape not supported %v", dims)
	}
	l.outData = make([]float32, num.Prod(dims))
	//log.Printf("viewLayer: %d output %v => %dx%d\n", layer, dims, width, height)
	l.outImage = image.NewNRGBA(image.Rect(0, 0, width, height))
}

func (l *viewLayer) addWeightImage(layer int, wDims, bDims []int) {
	if len(bDims) != 1 || len(wDims) < 1 || bDims[0] != wDims[len(wDims)-1] {
		log.Printf("viewLayer %d: weight shape not supported %v %v", layer, wDims, bDims)
		return
	}
	l.wShape, l.bShape = wDims, bDims
	switch len(wDims) {
	case 2:
		// fully connected layer
		l.wiy, l.wix = factorise(wDims[0], 0, 1)
		l.woy, l.wox = factorise(wDims[1], factorMinWeights, aspectWeights)
		l.wborder = 1
	case 4:
		// convolutional layer
		l.wix, l.wiy = wDims[0], wDims[1]*wDims[2]
		if wDims[2] == 1 {
			l.woy, l.wox = factorise(wDims[3], factorMinWeights, aspectWeights)
		} else {
			l.woy, l.wox = 1, wDims[3]
		}
		l.wborder = 2
	default:
		log.Printf("viewLayer %d: weight shape not supported %v %v", layer, wDims, bDims)
		return
	}
	l.wData = make([]float32, num.Prod(wDims))
	l.bData = make([]float32, num.Prod(bDims))
	//log.Printf("viewLayer: %d %v %v in=%dx%d out=%dx%d\n", layer, wDims, bDims, l.wix, l.wiy, l.wox, l.woy)
	l.wImage = image.NewNRGBA(image.Rect(0, 0, (l.wix+l.wborder)*l.wox, (l.wiy+l.wborder)*l.woy))
}

func (l *viewLayer) block(i int) (x, y int) {
	x = (l.wix + l.wborder) * (i % l.wox)
	y = (l.wiy + l.wborder) * (i / l.wox)
	return
}

// if n > nmin returns f1, f2 where f1*f2 = n and f1 <= aspect * f2 else 1, n
func factorise(n, nmin int, aspect float64) (f1, f2 int) {
	if n < 1 {
		panic("factorise: input must be >= 1")
	}
	if n > nmin {
		for f1 = int(math.Sqrt(float64(n) * aspect)); f1 > 1; f1-- {
			if n%f1 == 0 {
				return f1, n / f1
			}
		}
	}
	return 1, n
}

// convert value in range cmin:cmax to interpolated color from cmap
func mapColor(val float32) color.NRGBA {
	var col [3]float32
	ncol := len(cmap)
	switch {
	case val <= cmin:
		col = cmap[0]
	case val >= cmax:
		col = cmap[ncol-1]
	default:
		vsc := float32(ncol-1) * (val - cmin) / (cmax - cmin)
		ix := int(vsc)
		fx := vsc - float32(ix)
		for i := range col {
			col[i] = cmap[ix][i]*(1-fx) + cmap[ix+1][i]*fx
		}
	}
	return color.NRGBA{uint8(col[0] * 255), uint8(col[1] * 255), uint8(col[2] * 255), 255}
}
