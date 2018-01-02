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
	aspectOutput     = 0.1
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
	Weights []float32
	Biases  []float32
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
	log.Println("load model:", model)
	var err error
	n.NetworkData, err = LoadNetwork(model, false)
	if err != nil {
		return nil, err
	}
	if err := n.Init(n.Conf); err != nil {
		return nil, err
	}
	if err := n.Import(); err != nil {
		return nil, err
	}
	return n, nil
}

// Initialise the network
func (n *Network) Init(conf nnet.Config) error {
	var err error
	log.Printf("init network: dataSet=%s useGPU=%v\n", conf.DataSet, conf.UseGPU)
	if n.view != nil {
		n.view.queue.Finish()
		n.queue.Finish()
	}
	if n.Network == nil || conf.UseGPU != n.Network.UseGPU {
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
	n.trainData = nnet.NewDataset(n.queue.Dev(), n.Data["train"], conf.DatasetConfig(false), n.rng)
	n.Network = nnet.New(n.queue, conf, n.trainData.BatchSize, n.trainData.Shape(), n.rng)
	if n.DebugLevel >= 1 {
		fmt.Println(n.Network)
	}
	n.test.Init(n.queue, conf, n.Data, n.testRng).Predict()
	n.Labels = make(map[string][]int32)
	for key, dset := range n.test.Data {
		n.Labels[key] = make([]int32, dset.Samples)
		dset.Label(seq(dset.Samples), n.Labels[key])
	}
	if d, ok := n.Data["test"]; ok {
		n.view = newViewData(n.queue.Dev(), "test", d, n.trans["test"], conf, n.testRng)
	} else if d, ok = n.Data["train"]; ok {
		n.view = newViewData(n.queue.Dev(), "train", d, nil, conf, n.testRng)
	} else {
		err = fmt.Errorf("Network init: no test or train data")
	}
	return err
}

// release allocated buffers
func (n *Network) release() {
	if n.Network != nil {
		n.Network.Release()
	}
	if n.test != nil {
		n.test.Release()
	}
	if n.trainData != nil {
		n.trainData.Release()
	}
	if n.view != nil {
		n.view.Release()
		n.view.input.Release()
	}
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
	log.Println("init weights")
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
	n.running = true
	n.stop = false
	go func() {
		n.queue.Profiling(n.Profile)
		acc := n.queue.NewArray(num.Float32)
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
				loss := nnet.TrainEpoch(net, n.trainData, acc)
				done = n.test.Test(net, epoch, loss, start)
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
		n.Unlock()
		log.Println("train: end - quit =", quit)
		if n.Profile {
			fmt.Printf("== Profile ==\n%s\n", n.queue.Profile())
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
	if done || quit {
		log.Println(n.test.Stats[epoch-1].String(n.test.Headers, true))
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
			log.Println("nextEpoch: error writing to websocket", err)
		}
	} else {
		log.Println("nextEpoch: websocket is not initialised")
	}
	// save state to disk
	n.Lock()
	n.Export()
	err := SaveNetwork(n.NetworkData, false)
	n.Unlock()
	if err != nil {
		log.Println("nextEpoch: error saving network:", err)
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
	if n.test.Net == nil || n.test.Net.Layers == nil {
		return
	}
	for i, layer := range n.test.Net.Layers {
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
			).Finish()
			n.Params = append(n.Params, d)
		}
	}
}

// Import current state after loading from file
func (n *Network) Import() error {
	n.test.Stats = n.Stats
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
		n.view.loadWeights(n.Network)
	}
	return nil
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
		plist[i] = fmt.Sprintf("%s=%v", tuneOptHtml[i], h.Conf.Get(p))
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
	input   num.Array
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
	v.Network = nnet.New(v.queue, conf, 1, v.inShape, rng)
	v.inData = make([]float32, num.Prod(v.inShape))
	if conf.FlattenInput {
		v.input = dev.NewArray(num.Float32, len(v.inData), 1)
	} else {
		v.input = dev.NewArray(num.Float32, append(v.inShape, 1)...)
	}
	for i, layer := range v.Layers {
		l := viewLayer{ltype: layer.Type()}
		// filter output layers
		if l.ltype != "maxPool" && l.ltype != "dropout" && l.ltype != "flatten" {
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
			scale := 2 / math.Sqrt(float64(num.Prod(layer.InShape())))
			l.cmapW = moreland.SmoothGreenRed()
			l.cmapW.SetMin(-scale)
			l.cmapW.SetMax(scale)
			l.cmapB = moreland.SmoothGreenRed()
			l.cmapB.SetMin(-1)
			l.cmapB.SetMax(1)
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
		W, B := v.Layers[iLayer].(nnet.ParamLayer).Params()
		v.queue.Call(
			num.Read(W, l.wData),
			num.Read(B, l.bData),
		).Finish()
		for out := 0; out < l.wox*l.woy; out++ {
			xb := (l.wix + l.wborder) * (out % l.wox)
			yb := (l.wiy + l.wborder) * (out / l.wox)
			// draw bias
			biasCol := mapColor(l.bData[out], l.cmapB)
			for j := 0; j < l.wix; j++ {
				l.wImage.Set(xb+j, yb, biasCol)
			}
			for j := 0; j < l.wiy; j++ {
				l.wImage.Set(xb, yb+j, biasCol)
			}
			// draw weights
			switch len(l.wShape) {
			case 2:
				nin := l.wShape[0]
				for j, val := range l.wData[out*nin : (out+1)*nin] {
					l.wImage.Set(xb+j/l.wiy+1, yb+j%l.wiy+1, mapColor(val, l.cmapW))
				}
			case 4:
				w, h, nin := l.wShape[0], l.wShape[1], l.wShape[2]
				base := out * nin * w * h
				for in := 0; in < nin; in++ {
					xb2 := xb + w*(in%l.wix2) + 1
					yb2 := yb + h*(in/l.wix2) + 1
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
		log.Printf("viewLayer: output shape not supported %v", dims)
	}
	l.outData = make([]float32, num.Prod(dims))
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
		// fully connected layer: wDims = [nIn, nOut]
		l.wiy, l.wix = factorise(wDims[0], 0, 1)
		l.woy, l.wox = factorise(wDims[1], factorMinWeights, aspectWeights)
		l.wborder = 1
	case 4:
		// convolutional layer: wDims = [w, h, nIn, nOut]
		l.woy, l.wox = factorise(wDims[3], factorMinWeights, aspectWeights)
		l.wiy2, l.wix2 = factorise(wDims[2], 0, 1)
		l.wix, l.wiy = l.wix2*wDims[0], l.wiy2*wDims[1]
		l.wborder = 2
	default:
		log.Printf("viewLayer %d: weight shape not supported %v %v", layer, wDims, bDims)
		return
	}
	l.wData = make([]float32, num.Prod(wDims))
	l.bData = make([]float32, num.Prod(bDims))
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
		log.Fatalf("error in colormap lookup: %s\n", err)
	}
	return col
}
