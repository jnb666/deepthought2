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
	Layer   int
	WeightV []uint32
	BiasV   []uint32
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
	n.view.loadWeights(n.Network)
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
		log.Printf("memory used: %s\n", nnet.FormatBytes(n.Memory()+n.test.Memory()+n.view.Memory()))

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
	n.view.loadWeights(n.Network)
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
	for i, layer := range n.Layers {
		if l, ok := layer.(nnet.ParamLayer); ok {
			W, B := l.Params()
			d := LayerData{Layer: i, WeightV: make([]uint32, num.Prod(W.Dims))}
			if B != nil {
				d.BiasV = make([]uint32, num.Prod(B.Dims))
				n.queue.Call(num.Read(B, d.BiasV))
			}
			n.queue.Call(num.Read(W, d.WeightV)).Finish()
			n.Params = append(n.Params, d)
		}
	}
}

// Import current state after loading from file
func (n *Network) Import() {
	n.test.Stats = n.Stats
	if n.Epoch == 0 {
		log.Println("import: init weights")
		n.InitWeights(n.rng)
	} else if n.Params != nil && len(n.Params) > 0 {
		log.Println("import weights")
		nlayers := len(n.Network.Layers)
		for _, p := range n.Params {
			if p.Layer >= nlayers {
				log.Fatalf("ERROR: layer %d import error: network has %d layers total", p.Layer, nlayers)
			}
			layer, ok := n.Network.Layers[p.Layer].(nnet.ParamLayer)
			if !ok {
				log.Fatalf("ERROR: layer %d import error: not a ParamLayer", p.Layer)
			}
			W, B := layer.Params()
			if B != nil {
				if size := num.Prod(B.Dims); size != len(p.BiasV) {
					log.Fatalf("ERROR: layer %d import error: bias size mismatch - have %d - expect %d",
						p.Layer, len(p.BiasV), size)
				}
				n.queue.Call(num.Write(B, p.BiasV))
			}
			if size := num.Prod(W.Dims); size != len(p.WeightV) {
				log.Fatalf("ERROR: layer %d import error: weight size mismatch - have %d - expect %d",
					p.Layer, len(p.WeightV), size)
			}
			n.queue.Call(num.Write(W, p.WeightV))
		}
		n.view.loadWeights(n.Network)
	} else {
		log.Fatalln("ERROR: no weights to import!")
	}
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
	layers  []viewLayer
	dset    string
	data    nnet.Data
	trans   *img.Transformer
	input   *num.Array
	inShape []int
	inData  []float32
	cmapOut palette.ColorMap
}

type viewLayer struct {
	ltype      string
	outShape   []int
	outData    []float32
	outImage   *image.NRGBA
	ox, oy     int
	wShape     []int
	bShape     []int
	wData      []float32
	bData      []float32
	wImage     *image.NRGBA
	batchNorm  bool
	wix, wiy   int
	wix2, wiy2 int
	wox, woy   int
	wborder    int
	cmapW      palette.ColorMap
	cmapB      palette.ColorMap
}

func newViewData(dev num.Device, dset string, data nnet.Data, trans *img.Transformer, conf nnet.Config, rng *rand.Rand) *viewData {
	v := &viewData{queue: dev.NewQueue(), dset: dset, data: data, trans: trans}
	v.inShape = data.Shape()
	v.Network = nnet.New(v.queue, conf, 1, v.inShape, false, rng)
	v.inData = make([]float32, num.Prod(v.inShape))
	v.input = dev.NewArray(num.Float32, append(v.inShape, 1)...)
	for i, layer := range v.Layers {
		l := viewLayer{ltype: layer.Type()}
		// filter output layers
		if l.ltype == "linear" || l.ltype == "conv" || l.ltype == "activation" {
			shape := layer.OutShape()
			l.outShape = shape[:len(shape)-1]
		}
		if l.ltype == "activation" {
			for prev := len(v.layers) - 1; prev >= 0; prev-- {
				lp := v.layers[prev]
				if (lp.ltype == "conv" || lp.ltype == "linear") && num.SameShape(lp.outShape, l.outShape) {
					l.ltype = lp.ltype + " " + layer.ToString()
					v.layers[prev].outShape = nil
					break
				}
			}
		}
		// allocate buffers and images for weights and biases
		if pLayer, ok := layer.(nnet.ParamLayer); ok {
			//log.Printf("param layer %d: %v %v\n", i, pLayer.FilterShape(), pLayer.BiasShape())
			W, B := pLayer.Params()
			// conv followed by batchnorm?
			var W2 *num.Array
			if next := v.Layers[i+1]; l.ltype == "conv" && next.Type() == "batchNorm" && B == nil {
				W2, B = next.(nnet.ParamLayer).Params()
				l.batchNorm = true
			}
			if l.ltype == "linear" || l.ltype == "conv" {
				l.addWeightImage(i, W, B, W2)
				scale := 2 / math.Sqrt(float64(num.Prod(layer.InShape())))
				l.cmapW = moreland.SmoothGreenRed()
				l.cmapW.SetMin(-scale)
				l.cmapW.SetMax(scale)
				l.cmapB = moreland.SmoothGreenRed()
				l.cmapB.SetMin(-1)
				l.cmapB.SetMax(1)
			}
		}
		v.layers = append(v.layers, l)
	}
	// allocate buffers and output images
	for i, l := range v.layers {
		if l.outShape != nil {
			v.layers[i].addOutputImage(i, l.outShape)
		}
	}
	v.cmapOut = moreland.BlackBody()
	v.cmapOut.SetMin(0)
	v.cmapOut.SetMax(1)
	return v
}

func (v *viewData) loadWeights(net *nnet.Network) {
	net.CopyTo(v.Network, true)
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
	v.Fprop(v.input, false)
	for iLayer, l := range v.layers {
		if l.outImage == nil {
			continue
		}
		v.queue.Call(
			num.Read(v.Layers[iLayer].Output(), l.outData),
		).Finish()
		switch len(l.outShape) {
		case 1:
			height := l.outImage.Bounds().Dy()
			for i, val := range l.outData {
				l.outImage.Set(i/height, i%height, mapColor(val, v.cmapOut))
			}
		case 3:
			bh, bw := l.outShape[0], l.outShape[1]
			for i := 0; i < l.ox*l.oy; i++ {
				xb := (bw + 1) * (i % l.ox)
				yb := (bh + 1) * (i / l.ox)
				for j := 0; j < bw*bh; j++ {
					col := mapColor(l.outData[i*bw*bh+j], v.cmapOut)
					l.outImage.Set(xb+j/bh+1, yb+j%bh+1, col)
				}
			}
		}
	}
}

// update weight and bias images
func (v *viewData) updateWeights() {
	for iLayer, l := range v.layers {
		if l.wImage == nil {
			continue
		}
		var W2 *num.Array
		W, B := v.Layers[iLayer].(nnet.ParamLayer).Params()
		if B != nil {
			v.queue.Call(num.Read(B, l.bData))
		} else if l.batchNorm {
			W2, B = v.Layers[iLayer+1].(nnet.ParamLayer).Params()
			v.queue.Call(
				num.Read(W2, l.bData),
				num.Read(B, l.bData[W2.Size():]),
			)
		}
		v.queue.Call(num.Read(W, l.wData)).Finish()
		for out := 0; out < l.wox*l.woy; out++ {
			xb := (l.wix + l.wborder) * (out % l.wox)
			yb := (l.wiy + l.wborder) * (out / l.wox)
			offset := 0
			if l.bData != nil {
				// draw bias
				biasCol1 := mapColor(l.bData[out], l.cmapB)
				biasCol2 := mapColor(l.bData[out], l.cmapB)
				if l.batchNorm {
					biasCol2 = mapColor(l.bData[W2.Size()+out], l.cmapB)
				}
				for j := 0; j < l.wix; j++ {
					l.wImage.Set(xb+j, yb, biasCol1)
				}
				for j := 0; j < l.wiy; j++ {
					l.wImage.Set(xb, yb+j, biasCol2)
				}
				offset = 1
			}
			// draw weights
			switch len(l.wShape) {
			case 2:
				nin := l.wShape[0]
				for j, val := range l.wData[out*nin : (out+1)*nin] {
					l.wImage.Set(xb+j/l.wiy+offset, yb+j%l.wiy+offset, mapColor(val, l.cmapW))
				}
			case 4:
				w, h, nin := l.wShape[0], l.wShape[1], l.wShape[2]
				base := out * nin * w * h
				for in := 0; in < nin; in++ {
					xb2 := xb + w*(in%l.wix2) + offset
					yb2 := yb + h*(in/l.wix2) + offset
					for j, val := range l.wData[base+in*w*h : base+(in+1)*w*h] {
						l.wImage.Set(xb2+j/h, yb2+j%h, mapColor(val, l.cmapW))
					}
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
		log.Fatalf("ERROR: viewLayer - output shape not supported %v", dims)
	}
	l.outData = make([]float32, num.Prod(dims))
	l.outImage = image.NewNRGBA(image.Rect(0, 0, width, height))
}

func (l *viewLayer) addWeightImage(layer int, W, B, W2 *num.Array) {
	l.wShape = W.Dims
	if len(l.wShape) < 1 || (B != nil && B.Size() != l.wShape[len(l.wShape)-1]) {
		log.Fatalf("ERROR: viewLayer %d: weight shape not supported %v %v", layer, l.wShape, l.bShape)
	}
	l.wData = make([]float32, num.Prod(l.wShape))
	l.wborder = 1
	if B != nil {
		if W2 != nil {
			l.bShape = []int{B.Size() + W2.Size()}
		} else {
			l.bShape = []int{B.Size()}
		}
		l.bData = make([]float32, num.Prod(l.bShape))
		l.wborder++
	}
	switch len(l.wShape) {
	case 2:
		// fully connected layer: l.wShape = [nIn, nOut]
		l.wiy, l.wix = factorise(l.wShape[0], 0, 1)
		l.woy, l.wox = factorise(l.wShape[1], factorMinWeights, aspectWeights)
	case 4:
		// convolutional layer: l.wShape = [w, h, nIn, nOut]
		l.woy, l.wox = factorise(l.wShape[3], factorMinWeights, aspectWeights)
		l.wiy2, l.wix2 = factorise(l.wShape[2], 0, 1)
		l.wix, l.wiy = l.wix2*l.wShape[0], l.wiy2*l.wShape[1]
	default:
		log.Fatalf("ERROR: viewLayer %d: weight shape not supported %v %v", layer, l.wShape, l.bShape)
	}
	l.wImage = image.NewNRGBA(image.Rect(0, 0, (l.wix+l.wborder)*l.wox, (l.wiy+l.wborder)*l.woy))
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
