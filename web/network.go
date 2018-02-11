// Package web has a web based interface we network training and visualisation.
package web

import (
	"encoding/gob"
	"fmt"
	"github.com/gorilla/websocket"
	"github.com/jnb666/deepthought2/img"
	"github.com/jnb666/deepthought2/nnet"
	"github.com/jnb666/deepthought2/num"
	"gonum.org/v1/plot/palette"
	"gonum.org/v1/plot/palette/moreland"
	"html/template"
	"image"
	"image/color"
	"log"
	"math"
	"math/rand"
	"os"
	"path"
	"strconv"
	"strings"
	"sync"
	"time"
)

const (
	aspectOutput     = 0.2
	aspectWeights    = 0.2
	factorMinOutput  = 20
	factorMinWeights = 20
)

var tuneOpts = []string{"Eta", "Lambda", "TrainBatch"}
var tuneOptHtml = []string{"&eta;", "&lambda;", "batch"}

// Network and associated training / test data and configuration
type Network struct {
	*NetworkData
	*nnet.Network
	Data      map[string]nnet.Data
	Labels    map[string][]int32
	trans     map[string]*img.Transformer
	test      *nnet.TestBase
	conn      *websocket.Conn
	trainData *nnet.Dataset
	queue     num.Queue
	rng       *rand.Rand
	testRng   *rand.Rand
	view      *viewData
	updated   bool
	running   bool
	stop      bool
	tuneMode  bool
	sync.Mutex
}

// Embedded structs used to persist state to file
type NetworkData struct {
	Model   string
	Conf    nnet.Config
	MaxRun  int
	Run     int
	Epoch   int
	Stats   []nnet.Stats
	Pred    map[string][]int32
	Params  []LayerData
	History []HistoryData
	Tuners  []TuneParams
}

type LayerData struct {
	LayerId string
	WeightV []uint32
}

type HistoryData struct {
	Stats nnet.Stats
	Conf  nnet.Config
}

type TuneParams struct {
	Name    string
	Values  []string
	Boolean bool
}

// Create a new network and load config from data given model name
func NewNetwork(model string) (*Network, error) {
	n := &Network{test: nnet.NewTestBase()}
	var err error
	n.NetworkData, err = LoadNetwork(model, false)
	if err != nil {
		return nil, err
	}
	if err := n.Init(n.Conf); err != nil {
		return nil, err
	}
	n.Import()
	return n, nil
}

// Initialise the network
func (n *Network) Init(conf nnet.Config) error {
	var err error
	log.Printf("init network: model=%s dataSet=%s useGPU=%v\n", n.Model, conf.DataSet, conf.UseGPU)
	if n.view != nil {
		log.Println("sync queue")
		n.view.queue.Finish()
		n.queue.Finish()
	}
	if n.queue == nil || n.Network == nil || conf.UseGPU != n.Network.UseGPU {
		log.Println("new queue")
		n.queue = num.NewDevice(conf.UseGPU).NewQueue()
	}
	if n.Network == nil || conf.RandSeed != n.Network.RandSeed {
		n.rng = nnet.SetSeed(conf.RandSeed)
		n.testRng = nnet.SetSeed(conf.RandSeed)
	}
	if n.Network == nil || conf.DataSet != n.Network.DataSet {
		if n.Data, err = nnet.LoadData(conf.DataSet); err != nil {
			return err
		}
		n.trans = make(map[string]*img.Transformer)
		for key, data := range n.Data {
			if d, ok := data.(*img.Data); ok {
				n.trans[key] = img.NewTransformer(d, img.NoTrans, img.ConvBoxBlur, n.testRng)
			}
		}
	}
	n.release()
	dev := n.queue.Dev()
	n.trainData = nnet.NewDataset(dev, n.Data["train"], conf.DatasetConfig(false), n.rng)
	n.Network = nnet.New(n.queue, conf, n.trainData.BatchSize, n.trainData.Shape(), true, n.rng)
	n.test.Init(n.queue, conf, n.Data, n.testRng).Predict(n.trainData)
	n.Labels = make(map[string][]int32)
	for key, dset := range n.test.Data {
		n.initLabels(key, dset)
	}
	n.initLabels("train", n.trainData)
	if d, ok := n.Data["test"]; ok {
		n.view = newViewData(dev, "test", d, n.trans["test"], conf, n.testRng)
	} else if d, ok = n.Data["train"]; ok {
		n.view = newViewData(dev, "train", d, nil, conf, n.testRng)
	} else {
		err = fmt.Errorf("Network init: no test or train data")
	}
	return err
}

func (n *Network) initLabels(key string, dset *nnet.Dataset) {
	n.Labels[key] = make([]int32, dset.Samples)
	dset.Label(seq(dset.Samples), n.Labels[key])
}

// release allocated buffers
func (n *Network) release() {
	if n.Network != nil {
		n.Network.Release()
		n.Network = nil
	}
	if n.test != nil {
		n.test.Release()
	}
	if n.trainData != nil {
		n.trainData.Release()
		n.trainData = nil
	}
	if n.view != nil {
		n.view.Release()
		n.view.input.Release()
		n.view = nil
	}
}

func (n *Network) Queue() num.Queue {
	return n.queue
}

// Initialise for new training run
func (n *Network) Start(conf nnet.Config, lock bool) error {
	if lock {
		n.Lock()
		defer n.Unlock()
	}
	if err := n.Init(conf); err != nil {
		return err
	}
	n.test.Reset()
	log.Println("start: init weights")
	n.InitWeights(n.rng)
	n.view.loadWeights(n.queue, n.Network)
	n.Epoch = 0
	n.updated = false
	return nil
}

// Perform training run
func (n *Network) Train(restart bool) error {
	log.Printf("train %s: restart=%v\n", n.Model, restart)
	runs := []nnet.Config{n.Conf}
	if n.tuneMode {
		runs = getRunConfig(n.Conf, n.Tuners)
	}
	n.MaxRun = len(runs)
	if restart {
		if n.Epoch != 0 || n.Run != 0 || n.updated {
			n.Run = 0
			if err := n.Start(runs[0], false); err != nil {
				return err
			}
		}
		n.Epoch = 1
	} else if n.Epoch > 0 {
		n.Epoch++
	}
	if n.Epoch == 0 || n.Epoch > n.MaxEpoch {
		return nil
	}
	log.Println(n.Network)
	n.running = true
	n.stop = false
	go func() {
		quit := false
		for n.Run < n.MaxRun && !quit {
			if n.Run > 0 {
				if err := n.Start(runs[n.Run], true); err != nil {
					log.Println(err)
					return
				}
				n.Epoch = 1
			}
			log.Printf("train run %d / %d epoch=%d\n", n.Run+1, len(runs), n.Epoch)
			n.trainData.SetTrans(n.Normalise, n.Distort)
			epoch := n.Epoch
			done := false
			epilogue := false
			net := n.Network
			for !done && !quit {
				if n.test.Epilogue() && !epilogue {
					log.Printf("training for %d extra epochs\n", net.ExtraEpochs)
					n.trainData.SetTrans(n.Normalise, false)
					epilogue = true
				}
				start := time.Now()
				loss, trainErr := nnet.TrainEpoch(net, n.trainData, epoch, n.test.Pred["train"])
				done = n.test.Test(net, epoch, loss, trainErr, start)
				if net.LogEvery == 0 || epoch%net.LogEvery == 0 || done {
					quit = n.nextEpoch(epoch, done)
				}
				epoch++
			}
			if !quit {
				n.Run++
			}
		}
		n.Lock()
		n.running = false
		n.stop = false
		nnet.MemoryProfile(n.Conf.MemProfile, n.Network, n.test.Network())
		if n.Conf.Profile {
			log.Print(n.queue.Profile())
		}
		// save state to disk
		n.Export()
		err := SaveNetwork(n.NetworkData, false)
		n.Unlock()
		if err != nil {
			log.Println("ERROR: error saving network:", err)
		}
		log.Println("train: end - quit =", quit)
		if n.Profile {
			log.Printf("== Profile ==\n%s\n", n.queue.Profile())
		}
	}()
	return nil
}

func (n *Network) nextEpoch(epoch int, done bool) bool {
	quit := false
	n.Lock()
	n.Epoch = epoch
	// check for interrupt
	if n.stop {
		n.stop = false
		n.running = false
		quit = true
	}
	s := n.test.Stats[epoch-1]
	log.Println(s.String(n.test.Headers))
	if done || quit {
		log.Printf("train time:%s  total:%s", nnet.FormatDuration(s.TrainTime), nnet.FormatDuration(s.Elapsed))
	}
	// update predictions for each image
	for key, pred := range n.test.Pred {
		if arr, ok := n.Pred[key]; !ok || len(arr) != len(pred) {
			n.Pred[key] = make([]int32, len(pred))
		}
		copy(n.Pred[key], pred)
	}
	// update visualisation
	n.view.loadWeights(n.queue, n.Network)
	// update history
	if done && !quit && len(n.test.Stats) > 0 {
		n.History = append(n.History, HistoryData{
			Stats: n.test.Stats[len(n.test.Stats)-1].Copy(),
			Conf:  n.Config.Copy(),
		})
	}
	n.Unlock()
	// notify via websocket
	if n.conn != nil {
		msg := []byte(strconv.Itoa(n.Run+1) + ":" + strconv.Itoa(epoch))
		err := n.conn.WriteMessage(websocket.TextMessage, msg)
		if err != nil {
			log.Println("ERROR: error writing to websocket", err)
		}
	} else {
		log.Println("ERROR: websocket is not initialised")
	}
	return quit
}

func (n *Network) heading() template.HTML {
	s := fmt.Sprintf(`%s: run <span id="run">%d</span>/%d  epoch <span id="epoch">%d</span>/%d`, n.Model, n.Run+1, n.MaxRun, n.Epoch, n.MaxEpoch)
	return template.HTML(s)
}

// Export current state prior to saving to file
func (n *Network) Export() {
	n.Stats = n.test.Stats
	n.Params = []LayerData{}
	nnet.ParamLayers("", n.Layers, func(desc string, p nnet.ParamLayer) {
		n.Params = append(n.Params, LayerData{LayerId: desc, WeightV: p.Export(n.queue)})
	})
	n.queue.Finish()
}

// Import current state after loading from file
func (n *Network) Import() {
	n.test.Stats = n.Stats
	if n.Epoch == 0 {
		log.Println("import: init weights")
		n.InitWeights(n.rng)
	} else if n.Params != nil && len(n.Params) > 0 {
		log.Println("import weights")
		ix := 0
		nnet.ParamLayers("", n.Layers, func(desc string, p nnet.ParamLayer) {
			id := n.Params[ix].LayerId
			if desc != id {
				log.Fatalf("ERROR: layer import mismatch: got %s expecting %s", desc, id)
			}
			p.Import(n.queue, n.Params[ix].WeightV)
			ix++
		})
	} else {
		log.Fatalln("ERROR: no weights to import!")
	}
	n.view.loadWeights(n.queue, n.Network)
}

// Encode data in gob format and save to file under nnet.DataDir
func SaveNetwork(data *NetworkData, reset bool) error {
	model := data.Model
	filePath := path.Join(nnet.DataDir, model+".net")
	if reset {
		if err := data.Conf.Save(model + ".conf"); err != nil {
			return err
		}
		os.Remove(filePath)
		return nil
	}
	if f, err := os.Create(filePath); err != nil {
		return err
	} else {
		defer f.Close()
		log.Println("saving network config to", model+".net")
		return gob.NewEncoder(f).Encode(*data)
	}
}

// Read back gob encoded data file, if not found or reset is set then load default config.
func LoadNetwork(model string, reset bool) (data *NetworkData, err error) {
	data = &NetworkData{
		Model:   model,
		MaxRun:  1,
		Stats:   []nnet.Stats{},
		Pred:    map[string][]int32{},
		Params:  []LayerData{},
		History: []HistoryData{},
	}
	if !reset {
		if err = loadGob(model+".net", data); err != nil {
			reset = true
		}
	}
	if reset {
		data.Conf, err = nnet.LoadConfig(model + ".conf")
	}
	tuners := []TuneParams{}
	for i, opt := range tuneOpts {
		val := data.Conf.Get(opt)
		_, isBool := val.(bool)
		vals := []string{fmt.Sprint(val)}
		if i < len(data.Tuners) {
			vals = data.Tuners[i].Values
		}
		tuners = append(tuners, TuneParams{Name: opt, Values: vals, Boolean: isBool})
	}
	data.Tuners = tuners
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

// For hyperparameter tuning, get config per run
func getRunConfig(conf nnet.Config, params []TuneParams) []nnet.Config {
	for _, p := range params {
		conf = setConfig(conf, p.Name, p.Values[0])
	}
	logConfig(conf)
	list := permute(conf, params, len(params)-1, []nnet.Config{conf})
	log.Printf("getRunConfig: runs=%d cases=%d\n", conf.TrainRuns, len(list))
	res := []nnet.Config{}
	for run := 0; run < conf.TrainRuns; run++ {
		res = append(res, list...)
	}
	return res
}

func permute(conf nnet.Config, params []TuneParams, n int, list []nnet.Config) []nnet.Config {
	if n < 0 {
		return list
	}
	for i, val := range params[n].Values {
		if i > 0 {
			conf = setConfig(conf, params[n].Name, val)
			logConfig(conf)
			list = append(list, conf)
		}
		list = permute(conf, params, n-1, list)
	}
	return list
}

func setConfig(c nnet.Config, name string, val string) nnet.Config {
	var err error
	c, err = c.SetString(name, val)
	if err != nil {
		panic(err)
	}
	return c
}

func logConfig(c nnet.Config) {
	var s string
	for _, name := range tuneOpts {
		s += fmt.Sprintf("%s=%v ", name, c.Get(name))
	}
	log.Println("getRunConfig:", s)
}

func tuneParams(h HistoryData) string {
	plist := make([]string, len(tuneOpts))
	for i, p := range tuneOpts {
		plist[i] = fmt.Sprintf("%s=%v<br>", tuneOptHtml[i], h.Conf.Get(p))
	}
	return strings.Join(plist, " ")
}

// Data used for network visualtion of weights and outputs
type viewData struct {
	*nnet.Network
	queue   num.Queue
	weights []weightLayer
	layers  []viewLayer
	dset    string
	data    nnet.Data
	trans   *img.Transformer
	input   *num.Array
	inShape []int
	inData  []float32
	cmapOut palette.ColorMap
	cmapB   palette.ColorMap
}

type weightLayer struct {
	desc     string
	wShape   []int
	bShape   []int
	wData    []float32
	bData    []float32
	ix, iy   int
	ix2, iy2 int
	ox, oy   int
	border   int
	image    *image.NRGBA
	cmapW    palette.ColorMap
}

type viewLayer struct {
	desc   string
	shape  []int
	data   []float32
	image  *image.NRGBA
	ox, oy int
}

func newViewData(dev num.Device, dset string, data nnet.Data, trans *img.Transformer, conf nnet.Config, rng *rand.Rand) *viewData {
	v := &viewData{
		queue:   dev.NewQueue(),
		dset:    dset,
		data:    data,
		trans:   trans,
		cmapOut: moreland.BlackBody(),
		cmapB:   moreland.SmoothGreenRed(),
	}
	v.inShape = data.Shape()
	v.Network = nnet.New(v.queue, conf, 1, v.inShape, false, rng)
	v.inData = make([]float32, num.Prod(v.inShape))
	v.input = dev.NewArray(num.Float32, append(v.inShape, 1)...)
	v.cmapOut.SetMin(0)
	v.cmapOut.SetMax(1)
	v.cmapB.SetMin(-1)
	v.cmapB.SetMax(1)
	// weight data
	nnet.ParamLayers("", v.Layers, func(desc string, l nnet.ParamLayer) {
		v.weights = append(v.weights, newWeightLayer(l))
	})
	// output data
	outLayers("", v.Layers, func(desc string, l nnet.Layer) {
		v.layers = append(v.layers, newViewLayer(desc, l.OutShape()))
	})
	return v
}

func (v *viewData) loadWeights(q num.Queue, net *nnet.Network) {
	nnet.CopyParams(q, net.Layers, v.Layers, true)
}

// update output images with given index from test set
func (v *viewData) updateOutputs(index int) {
	if v.trans != nil {
		if v.Normalise {
			v.trans.Trans = img.Normalise
		} else {
			v.trans.Trans = img.NoTrans
		}
	}
	v.data.Input([]int{index}, v.inData, v.trans)
	v.queue.Call(
		num.Write(v.input, v.inData),
	)
	nnet.Fprop(v.queue, v.Layers, v.input, v.WorkSpace[0], false)
	ix := 0
	outLayers("", v.Layers, func(desc string, layer nnet.Layer) {
		l := v.layers[ix]
		v.queue.Call(
			num.Read(layer.Output(), l.data),
		).Finish()
		switch len(l.shape) {
		case 2:
			height := l.image.Bounds().Dy()
			for i, val := range l.data {
				l.image.Set(i/height, i%height, mapColor(val, v.cmapOut))
			}
		case 4:
			bh, bw := l.shape[0], l.shape[1]
			for i := 0; i < l.ox*l.oy; i++ {
				xb := (bw + 1) * (i % l.ox)
				yb := (bh + 1) * (i / l.ox)
				for j := 0; j < bw*bh; j++ {
					col := mapColor(l.data[i*bw*bh+j], v.cmapOut)
					l.image.Set(xb+j/bh+1, yb+j%bh+1, col)
				}
			}
		}
		ix++
	})
}

// update weight and bias images
func (v *viewData) updateWeights() {
	ix := 0
	nnet.ParamLayers("", v.Layers, func(desc string, p nnet.ParamLayer) {
		l := v.weights[ix]
		W, B := p.Params()
		v.queue.Call(num.Read(W, l.wData))
		if B != nil {
			if p.Type() == "batchNorm" {
				v.queue.Call(num.Read(B, l.wData[W.Size():]))
			} else {
				v.queue.Call(num.Read(B, l.bData))
			}
		}
		v.queue.Finish()
		for out := 0; out < l.ox*l.oy; out++ {
			xb := (l.ix + l.border) * (out % l.ox)
			yb := (l.iy + l.border) * (out / l.ox)
			offset := 0
			if l.bData != nil {
				// draw bias
				biasCol1 := mapColor(l.bData[out], v.cmapB)
				biasCol2 := mapColor(l.bData[out], v.cmapB)
				for j := 0; j < l.ix; j++ {
					l.image.Set(xb+j, yb, biasCol1)
				}
				for j := 0; j < l.iy; j++ {
					l.image.Set(xb, yb+j, biasCol2)
				}
				offset = 1
			}
			// draw weights
			switch p.Type() {
			case "linear":
				nin := l.wShape[0]
				for j, val := range l.wData[out*nin : (out+1)*nin] {
					l.image.Set(xb+j/l.iy+offset, yb+j%l.iy+offset, mapColor(val, l.cmapW))
				}
			case "conv":
				w, h, nin := l.wShape[0], l.wShape[1], l.wShape[2]
				base := out * nin * w * h
				for in := 0; in < nin; in++ {
					xb2 := xb + w*(in%l.ix2) + offset
					yb2 := yb + h*(in/l.ix2) + offset
					for j, val := range l.wData[base+in*w*h : base+(in+1)*w*h] {
						l.image.Set(xb2+j/h, yb2+j%h, mapColor(val, l.cmapW))
					}
				}
			case "batchNorm":
				for j, val := range l.wData {
					l.image.Set(j, 0, mapColor(val, v.cmapB))
				}
			}
		}
		ix++
	})
}

func (v *viewData) lastLayer() *viewLayer {
	if len(v.layers) == 0 {
		return nil
	}
	return &v.layers[len(v.layers)-1]
}

// allocate buffers and images for outputs
func newViewLayer(desc string, dims []int) viewLayer {
	l := viewLayer{
		desc:  fmt.Sprintf("%s %v", desc, dims[:len(dims)-1]),
		shape: dims,
		data:  make([]float32, num.Prod(dims)),
	}
	var width, height int
	switch len(dims) {
	case 2:
		// fully connected layer
		height, width = factorise(dims[0], factorMinOutput, aspectOutput)
	case 4:
		// convolutional layer
		l.oy, l.ox = factorise(dims[2], factorMinOutput, aspectOutput)
		height = (dims[1] + 1) * l.oy
		width = (dims[0] + 1) * l.ox
	default:
		log.Fatalf("ERROR: viewLayer - output shape not supported %v", dims)
	}
	l.image = image.NewNRGBA(image.Rect(0, 0, width, height))
	//log.Println("add output: ", l.desc, l.image.Bounds())
	return l
}

// allocate buffers and images for weights and biases
func newWeightLayer(layer nnet.ParamLayer) weightLayer {
	W, B := layer.Params()
	l := weightLayer{
		wShape: W.Dims,
		wData:  make([]float32, W.Size()),
		border: 1,
		cmapW:  moreland.SmoothGreenRed(),
	}
	switch layer.Type() {
	case "linear":
		// fully connected layer: [nIn, nOut]
		l.iy, l.ix = factorise(l.wShape[0], 0, 1)
		l.oy, l.ox = factorise(l.wShape[1], factorMinWeights, aspectWeights)
	case "conv":
		// convolutional layer: [w, h, nIn, nOut]
		l.oy, l.ox = factorise(l.wShape[3], factorMinWeights, aspectWeights)
		l.iy2, l.ix2 = factorise(l.wShape[2], 0, 1)
		l.ix, l.iy = l.ix2*l.wShape[0], l.iy2*l.wShape[1]
	case "batchNorm":
		// batch norm layer: [nIn, 2]
		if B != nil {
			l.wShape = []int{W.Size(), 2}
			l.wData = append(l.wData, make([]float32, W.Size())...)
			B = nil
		}
		l.border = 0
		l.ox, l.oy = 1, 1
		l.ix, l.iy = 2*l.wShape[0], 1
	}
	l.desc = fmt.Sprintf("%s %v", layer.String(), l.wShape)
	if B != nil {
		l.desc += fmt.Sprintf(" %v", B.Dims)
		l.bShape = []int{B.Size()}
		l.bData = make([]float32, B.Size())
		l.border++
	}
	l.image = image.NewNRGBA(image.Rect(0, 0, (l.ix+l.border)*l.ox, (l.iy+l.border)*l.oy))
	scale := 2 / math.Sqrt(float64(num.Prod(layer.InShape())))
	l.cmapW.SetMin(-scale)
	l.cmapW.SetMax(scale)
	//log.Println("add weights: ", l.desc, l.image.Bounds())
	return l
}

// loop over output layers
func outLayers(desc string, layers []nnet.Layer, callback func(desc string, layer nnet.Layer)) {
	for _, l := range layers {
		switch l.Type() {
		case "pool":
			callback(l.Type(), l)
		case "activation":
			callback(desc+" "+l.String(), l)
		case "batchNorm":
			desc += " batchNorm"
		case "linear", "conv":
			desc = l.Type()
		}
		if group, ok := l.(nnet.LayerGroup); ok {
			outLayers(desc, group.Layers(), callback)
		}
	}
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
func mapColor(val float32, cmap palette.ColorMap) color.Color {
	v := float64(val)
	if v < cmap.Min() {
		v = cmap.Min()
	}
	if v > cmap.Max() {
		v = cmap.Max()
	}
	col, err := cmap.At(v)
	if err != nil {
		log.Fatalf("ERROR: error in colormap lookup for %s\n", err)
	}
	return col
}
