package nnet

import (
	"encoding/gob"
	"fmt"
	_ "github.com/jnb666/deepthought2/img"
	"github.com/jnb666/deepthought2/num"
	"image"
	"io"
	"math/rand"
	"os"
	"path"
	"strconv"
	"sync"
)

var (
	DataDir   = os.Getenv("GOPATH") + "/src/github.com/jnb666/deepthought2/data"
	DataTypes = []string{"train", "test", "valid"}
)

func init() {
	gob.Register(data{})
}

// Data interface type represents the raw data for a training or test set
type Data interface {
	Len() int
	Classes() []string
	Shape() []int
	Label(index []int, label []int32)
	Input(index []int, buf []float32)
	Image(i int) image.Image
	SetFile(io.ReadSeeker)
	File() io.ReadSeeker
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
	sync.WaitGroup
}

// Create a new Dataset struct, allocate array buffers  and set the batch size and maxSamples
func NewDataset(dev num.Device, data Data, batchSize, maxSamples int, flattenInput bool, rng *rand.Rand) *Dataset {
	d := &Dataset{Data: data, Samples: data.Len(), rng: rng}
	if maxSamples > 0 && d.Samples > maxSamples {
		d.Samples = maxSamples
	}
	if batchSize == 0 || batchSize > d.Samples {
		d.BatchSize = d.Samples
	} else {
		d.BatchSize = batchSize
	}
	d.Batches = d.Samples / d.BatchSize
	if d.Samples%d.BatchSize != 0 {
		d.Batches++
	}
	nfeat := num.Prod(data.Shape())
	d.xBuffer = make([]float32, nfeat*d.BatchSize)
	d.yBuffer = make([]int32, d.BatchSize)
	for i := range d.x {
		if flattenInput {
			d.x[i] = dev.NewArray(num.Float32, nfeat, d.BatchSize)
		} else {
			d.x[i] = dev.NewArray(num.Float32, append(data.Shape(), d.BatchSize)...)
		}
		d.y[i] = dev.NewArray(num.Int32, d.BatchSize)
		d.y1H[i] = dev.NewArray(num.Float32, len(d.Classes()), d.BatchSize)
	}
	d.indexes = make([]int, d.Samples)
	for i := range d.indexes {
		d.indexes[i] = i
	}
	d.queue = dev.NewQueue()
	return d
}

// release allocated buffers
func (d *Dataset) Release() {
	d.Wait()
	for i := range d.x {
		d.x[i].Release()
		d.y[i].Release()
		d.y1H[i].Release()
	}
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
		d.Input(d.indexes[start:end], d.xBuffer)
		d.Label(d.indexes[start:end], d.yBuffer)
		d.queue.Call(
			num.Write(d.x[d.buf], d.xBuffer),
			num.Write(d.y[d.buf], d.yBuffer),
			num.Onehot(d.y[d.buf], d.y1H[d.buf], len(d.Classes())),
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

// Rewind to start of data
func (d *Dataset) Rewind() {
	d.Wait()
	d.epoch = 0
	d.batch = 0
	d.loadBatch()
}

// Called at start of each epoch
func (d *Dataset) NextEpoch() {
	d.Wait()
	d.epoch++
	d.batch = 0
	if f := d.File(); f != nil && d.epoch > 1 {
		err := gob.NewDecoder(f).Decode(&d.Data)
		if err == io.EOF {
			fmt.Println("rewind file")
			f.Seek(0, io.SeekStart)
			err = gob.NewDecoder(f).Decode(&d.Data)
		}
		if err != nil {
			panic(err)
		}
		d.SetFile(f)
	}
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
		return nil, err
	}
	fmt.Printf("loading data from %s.dat:\t", name)
	var d Data
	if err = gob.NewDecoder(f).Decode(&d); err != nil {
		return nil, err
	}
	d.SetFile(f)
	fmt.Println(append(d.Shape(), d.Len()))
	return d, nil
}

// Encode in gob format and save to file under DataDir
func SaveDataFile(d Data, name string, append bool) error {
	filePath := path.Join(DataDir, name+".dat")
	var flags int
	if append {
		flags = os.O_WRONLY | os.O_CREATE | os.O_APPEND
	} else {
		flags = os.O_WRONLY | os.O_CREATE | os.O_TRUNC
	}
	f, err := os.OpenFile(filePath, flags, 0644)
	if err != nil {
		return err
	}
	defer f.Close()
	if !append {
		fmt.Println("saving data to", name+".dat")
	}
	return gob.NewEncoder(f).Encode(&d)
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
func NewData(nclasses int, shape []int, labels []int32, inputs []float32) data {
	classes := make([]string, nclasses)
	for i := range classes {
		classes[i] = strconv.Itoa(i)
	}
	return data{Class: classes, Dims: shape, Labels: labels, Inputs: inputs}
}

func (d data) Len() int { return len(d.Labels) }

func (d data) Classes() []string { return d.Class }

func (d data) Shape() []int { return d.Dims }

func (d data) Label(index []int, label []int32) {
	for i, ix := range index {
		label[i] = d.Labels[ix]
	}
}

func (d data) Input(index []int, buf []float32) {
	nfeat := num.Prod(d.Dims)
	for i, ix := range index {
		copy(buf[i*nfeat:], d.Inputs[ix*nfeat:(ix+1)*nfeat])
	}
}

func (d data) Image(i int) image.Image { return nil }

func (d data) SetFile(io.ReadSeeker) {}

func (d data) File() io.ReadSeeker { return nil }
