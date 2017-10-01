package nnet

import (
	"encoding/gob"
	"fmt"
	_ "github.com/jnb666/deepthought2/img"
	"github.com/jnb666/deepthought2/num"
	"image"
	"math/rand"
	"os"
	"path"
	"sync"
)

var (
	AccelConv = true
	DataDir   = os.Getenv("GOPATH") + "/src/github.com/jnb666/deepthought2/data"
	DataTypes = []string{"train", "test", "valid"}
)

func init() {
	gob.Register(data{})
}

// Data interface type represents the raw data for a training or test set
type Data interface {
	Len() int
	Classes() int
	Shape() []int
	Label(index []int, label []int32)
	Input(index []int, buf []float32)
	Image(i int) image.Image
	Distort(amount float64, rng *rand.Rand, accel bool)
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
	nfeat     int
	batch     int
	rng       *rand.Rand
	sync.WaitGroup
}

// Create a new Dataset struct, allocate array buffers  and set the batch size and maxSamples
func NewDataset(dev num.Device, data Data, batchSize, maxSamples int, distort float64, rng *rand.Rand) *Dataset {
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
		d.x[i] = dev.NewArray(num.Float32, nfeat, d.BatchSize)
		d.y[i] = dev.NewArray(num.Int32, d.BatchSize)
		d.y1H[i] = dev.NewArray(num.Float32, d.Classes(), d.BatchSize)
	}
	d.indexes = make([]int, d.Samples)
	for i := range d.indexes {
		d.indexes[i] = i
	}
	d.queue = dev.NewQueue()
	if distort > 0 {
		d.Distort(distort, rng, AccelConv)
	}
	return d
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
			num.Onehot(d.y[d.buf], d.y1H[d.buf], d.Classes()),
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
	d.batch = 0
	d.loadBatch()
}

// Shuffle the data set
func (d *Dataset) Shuffle() {
	d.indexes = d.rng.Perm(d.Samples)
}

// Load data from disk given the model name.
func LoadData(model string) (map[string]Data, error) {
	d := make(map[string]Data)
	for _, key := range DataTypes {
		name := model + "_" + key
		if FileExists(name + ".dat") {
			data, err := LoadDataFile(name)
			if err != nil {
				return nil, err
			}
			d[key] = data
		}
	}
	return d, nil
}

// Decode data from file in gob format under DataDir
func LoadDataFile(name string) (d Data, err error) {
	var f *os.File
	filePath := path.Join(DataDir, name+".dat")
	if f, err = os.Open(filePath); err != nil {
		return
	}
	defer f.Close()
	fmt.Printf("loading data from %s.dat:\t", name)
	if err = gob.NewDecoder(f).Decode(&d); err != nil {
		return
	}
	fmt.Println(append(d.Shape(), d.Len()))
	return
}

// Encode in gob format and save to file under DataDir
func SaveDataFile(d Data, name string) error {
	filePath := path.Join(DataDir, name+".dat")
	f, err := os.Create(filePath)
	if err != nil {
		return err
	}
	defer f.Close()
	fmt.Println("saving data to", name+".dat")
	return gob.NewEncoder(f).Encode(&d)
}

// Check if file exists under DataDir
func FileExists(name string) bool {
	filePath := path.Join(DataDir, name)
	_, err := os.Stat(filePath)
	return err == nil
}

type data struct {
	NClass int
	Dims   []int
	Labels []int32
	Inputs []float32
}

// NewData function creates a new data set which implements the Data interface
func NewData(classes int, shape []int, labels []int32, inputs []float32) data {
	return data{NClass: classes, Dims: shape, Labels: labels, Inputs: inputs}
}

func (d data) Len() int { return len(d.Labels) }

func (d data) Classes() int { return d.NClass }

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

func (d data) Distort(amount float64, rng *rand.Rand, accel bool) {}
