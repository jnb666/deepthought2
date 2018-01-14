package nnet

import (
	"encoding/gob"
	"fmt"
	"github.com/jnb666/deepthought2/img"
	"github.com/jnb666/deepthought2/num"
	"io"
	"math/rand"
	"os"
	"path"
	"strconv"
	"strings"
	"sync"
)

const headerBytes = 16

var (
	DataDir   = os.Getenv("GOPATH") + "/src/github.com/jnb666/deepthought2/data"
	DataTypes = []string{"train", "test", "valid"}
)

func init() {
	gob.Register(&data{})
}

// Data interface type represents the raw data for a training or test set
type Data interface {
	Len() int
	Classes() []string
	ClassSize() int
	Shape() []int
	Label(index []int, label []int32)
	Input(index []int, buf []float32, t *img.Transformer)
	Image(ix int, channel string) *img.Image
	Encode(w io.Writer) error
	Decode(r io.Reader) error
}

// Dataset type encapsulates a set of training, test or validation data.
type Dataset struct {
	Data
	Samples   int
	BatchSize int
	Batches   int
	queue     num.Queue
	xBuffer   []float32
	yBuffer   []int32
	x, y, y1H [2]num.Array
	indexes   []int
	buf       int
	epoch     int
	batch     int
	rng       *rand.Rand
	trans     *img.Transformer
	sync.WaitGroup
}

// Config options for dataset
type DatasetOptions struct {
	BatchSize  int
	MaxSamples int
	Normalise  bool
	Distort    bool
}

// Create a new Dataset struct, allocate array buffers  and set the batch size and maxSamples
func NewDataset(dev num.Device, data Data, opts DatasetOptions, rng *rand.Rand) *Dataset {
	d := &Dataset{Data: data, Samples: data.Len(), rng: rng}
	if opts.MaxSamples > 0 && d.Samples > opts.MaxSamples {
		d.Samples = opts.MaxSamples
	}
	if opts.BatchSize == 0 || opts.BatchSize > d.Samples {
		d.BatchSize = d.Samples
	} else {
		d.BatchSize = opts.BatchSize
	}
	d.Batches = d.Samples / d.BatchSize
	if d.Samples%d.BatchSize != 0 {
		d.Batches++
	}
	d.SetTrans(opts.Normalise, opts.Distort)
	nfeat := num.Prod(data.Shape())
	d.xBuffer = make([]float32, nfeat*d.BatchSize)
	d.yBuffer = make([]int32, d.BatchSize)
	for i := range d.x {
		d.x[i] = dev.NewArray(num.Float32, append(data.Shape(), d.BatchSize)...)
		d.y[i] = dev.NewArray(num.Int32, d.BatchSize)
		d.y1H[i] = dev.NewArray(num.Float32, d.ClassSize(), d.BatchSize)
	}
	d.indexes = make([]int, d.Samples)
	for i := range d.indexes {
		d.indexes[i] = i
	}
	d.queue = dev.NewQueue()
	return d
}

// Enable profiling for the associated queue
func (d *Dataset) Profiling(on bool, title string) {
	d.queue.Profiling(on, title)
}

// Set image transform
func (d *Dataset) SetTrans(normalise, distort bool) {
	if imgData, ok := d.Data.(*img.Data); ok {
		trans := imgData.Images[0].TransformType(normalise, distort)
		d.trans = img.NewTransformer(imgData, trans, img.ConvBoxBlur, d.rng)
	}
}

// Transforms applied to input data
func (d *Dataset) Trans() img.TransType {
	if d.trans == nil {
		return 0
	}
	return d.trans.Trans
}

// release allocated buffers
func (d *Dataset) Release() {
	d.Wait()
	for i := range d.x {
		d.x[i].Release()
		d.y[i].Release()
		d.y1H[i].Release()
	}
	d.queue.Shutdown()
}

// kick of load of next batch of data in background
func (d *Dataset) loadBatch() {
	d.Add(1)
	go func() {
		start := d.batch * d.BatchSize
		end := start + d.BatchSize
		if end > d.Samples {
			end = d.Samples
		}
		d.Input(d.indexes[start:end], d.xBuffer, d.trans)
		d.Label(d.indexes[start:end], d.yBuffer)
		d.queue.Call(
			num.Write(d.x[d.buf], d.xBuffer),
			num.Write(d.y[d.buf], d.yBuffer),
			num.Onehot(d.y[d.buf], d.y1H[d.buf], d.ClassSize()),
		)
		d.queue.Finish()
		d.Done()
	}()
}

// Get next batch of data
func (d *Dataset) NextBatch() (x, y, yOneHot num.Array) {
	d.Wait()
	x, y, yOneHot = d.x[d.buf], d.y[d.buf], d.y1H[d.buf]
	d.batch = (d.batch + 1) % d.Batches
	d.buf = (d.buf + 1) % 2
	d.loadBatch()
	return
}

// Called at start of each epoch
func (d *Dataset) NextEpoch() {
	d.Wait()
	d.batch = 0
	d.loadBatch()
}

// Shuffle the data set
func (d *Dataset) Shuffle() {
	d.indexes = d.rng.Perm(d.Samples)
}

// Load data from disk given the model name.
func LoadData(model string) (d map[string]Data, err error) {
	var data Data
	d = make(map[string]Data)
	for _, key := range DataTypes {
		name := model + "_" + key
		if FileExists(name + ".dat") {
			if data, err = LoadDataFile(name); err != nil {
				return
			}
			d[key] = data
		}
	}
	return d, nil
}

// Decode data from file in gob format under DataDir
func LoadDataFile(name string) (Data, error) {
	filePath := path.Join(DataDir, name+".dat")
	f, err := os.Open(filePath)
	if err != nil {
		return nil, fmt.Errorf("Error opening file %s.dat: %s", name, err)
	}
	buf := make([]byte, headerBytes)
	if n, err := f.Read(buf); err != nil || n != headerBytes {
		return nil, fmt.Errorf("Error reading file %s.dat: %s", name, err)
	}
	dtype := strings.TrimSpace(string(buf))
	var d Data
	switch dtype {
	case "*nnet.data":
		d = &data{}
	case "*img.Data":
		d = &img.Data{}
	default:
		return nil, fmt.Errorf("Error reading file %s.dat: invalid type header %s", name, dtype)
	}
	fmt.Printf("loading data from %s.dat\t", name)
	if err = d.Decode(f); err != nil {
		return nil, fmt.Errorf("Error decoding file %s.dat: %s", name, err)
	}
	fmt.Println(append(d.Shape(), d.Len()))
	return d, nil
}

// Encode in gob format and save to file under DataDir
func SaveDataFile(d Data, name string) error {
	filePath := path.Join(DataDir, name+".dat")
	f, err := os.Create(filePath)
	if err != nil {
		return err
	}
	defer f.Close()
	fmt.Printf("saving data to %s.dat\n", name)
	fmt.Fprintf(f, "%-"+strconv.Itoa(headerBytes)+"T", d)
	return d.Encode(f)
}

// Check if file exists under DataDir
func FileExists(name string) bool {
	filePath := path.Join(DataDir, name)
	_, err := os.Stat(filePath)
	return err == nil
}

type data struct {
	Class  []string
	Dims   []int
	Labels []int32
	Inputs []float32
}

// NewData function creates a new data set which implements the Data interface
func NewData(classes []string, shape []int, labels []int32, inputs []float32) *data {
	return &data{Class: classes, Dims: shape, Labels: labels, Inputs: inputs}
}

func (d *data) Len() int { return len(d.Labels) }

func (d *data) Classes() []string { return d.Class }

func (d *data) ClassSize() int {
	if len(d.Class) > 2 {
		return len(d.Class)
	}
	return 1
}

func (d *data) Shape() []int { return d.Dims }

func (d *data) Label(index []int, label []int32) {
	for i, ix := range index {
		label[i] = d.Labels[ix]
	}
}

func (d *data) Input(index []int, buf []float32, t *img.Transformer) {
	nfeat := num.Prod(d.Dims)
	for i, ix := range index {
		copy(buf[i*nfeat:], d.Inputs[ix*nfeat:(ix+1)*nfeat])
	}
}

func (d *data) Image(ix int, channel string) *img.Image {
	nfeat := num.Prod(d.Dims)
	img := img.NewImage(1, nfeat, 1)
	copy(img.Pix, d.Inputs[ix*nfeat:(ix+1)*nfeat])
	return img
}

func (d *data) Encode(w io.Writer) error {
	return gob.NewEncoder(w).Encode(*d)
}

func (d *data) Decode(r io.Reader) error {
	return gob.NewDecoder(r).Decode(d)
}
