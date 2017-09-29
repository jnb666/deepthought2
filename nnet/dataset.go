package nnet

import (
	"github.com/jnb666/deepthought2/num"

	"encoding/gob"
	"fmt"
	"image"
	"image/color"
	"math/rand"
	"os"
	"path"
	"sync"
)

var (
	DataDir   = os.Getenv("GOPATH") + "/src/github.com/jnb666/deepthought2/data"
	DataTypes = []string{"train", "test", "valid"}
)

// Dataset type encapsulates a set of training, test or validation data.
type Dataset struct {
	Data
	Samples   int
	BatchSize int
	Batches   int
	queue     num.Queue
	x, y, y1H [2]num.Array
	indexes   []int
	shuffled  bool
	buf       int
	start     int
	sync.WaitGroup
}

// Create a new Dataset struct, allocate array buffers  and set the batch size and maxSamples
func NewDataset(dev num.Device, data Data, batchSize, maxSamples int) *Dataset {
	d := &Dataset{Data: data, Samples: len(data.Labels)}
	if maxSamples > 0 && d.Samples > maxSamples {
		d.Samples = maxSamples
	}
	if len(d.Input) < d.Nfeat()*d.Samples {
		panic(fmt.Sprintf("invalid input size: have %d expecting %d x %d", len(d.Input), d.Nfeat(), d.Samples))
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
	for i := range d.x {
		d.x[i] = dev.NewArray(num.Float32, d.Nfeat(), d.BatchSize)
		d.y[i] = dev.NewArray(num.Int32, d.BatchSize)
		d.y1H[i] = dev.NewArray(num.Float32, d.Classes, d.BatchSize)
	}
	d.queue = dev.NewQueue()
	return d
}

// kick of load of next batch of data in background
func (d *Dataset) loadBatch() {
	d.Add(1)
	go func() {
		nfeat := d.Nfeat()
		end := d.start + d.BatchSize
		if end > d.Samples {
			end = d.Samples
		}
		if d.shuffled {
			for i, ix := range d.indexes[d.start:end] {
				d.queue.Call(
					num.WriteCol(d.x[d.buf], i, d.Input[ix*nfeat:(ix+1)*nfeat]),
					num.WriteCol(d.y[d.buf], i, &d.Labels[ix]),
				)
			}
		} else {
			d.queue.Call(
				num.Write(d.x[d.buf], d.Input[d.start*nfeat:end*nfeat]),
				num.Write(d.y[d.buf], d.Labels[d.start:end]),
			)
		}
		d.queue.Call(
			num.Onehot(d.y[d.buf], d.y1H[d.buf], d.Classes),
		)
		d.queue.Finish()
		d.Done()
	}()
}

// Get next batch of data
func (d *Dataset) NextBatch() (x, y, yOneHot num.Array) {
	d.Wait()
	x, y, yOneHot = d.x[d.buf], d.y[d.buf], d.y1H[d.buf]
	d.start += d.BatchSize
	if d.start > d.Samples {
		d.start = 0
	}
	d.buf = (d.buf + 1) % 2
	d.loadBatch()
	return
}

// Rewind to start of data
func (d *Dataset) Rewind() {
	d.start = 0
	d.buf = 0
	d.loadBatch()
}

// Shuffle the data set
func (d *Dataset) Shuffle(rng *rand.Rand) {
	d.indexes = rng.Perm(d.Samples)
	d.shuffled = true
}

// Data type has the raw data for a training or test set
type Data struct {
	Classes int
	Shape   []int
	Input   []float32
	Labels  []int32
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
func LoadDataFile(name string) (Data, error) {
	var d Data
	filePath := path.Join(DataDir, name+".dat")
	f, err := os.Open(filePath)
	if err != nil {
		return d, err
	}
	defer f.Close()
	fmt.Printf("loading data from %s.dat:\t", name)
	dec := gob.NewDecoder(f)
	err = dec.Decode(&d)
	fmt.Println(append(d.Shape, len(d.Labels)))
	return d, err
}

// Encode in gob format and save to file under DataDir
func (d Data) Save(name string) error {
	filePath := path.Join(DataDir, name+".dat")
	f, err := os.Create(filePath)
	if err != nil {
		return err
	}
	defer f.Close()
	fmt.Println("saving data to", name+".dat")
	enc := gob.NewEncoder(f)
	return enc.Encode(&d)
}

// Number of input features
func (d Data) Nfeat() int {
	return num.Prod(d.Shape)
}

// Export ith value from the input as an image
func (d Data) Image(i int, col color.NRGBA) *image.NRGBA {
	h, w := d.Shape[0], d.Shape[1]
	img := image.NewNRGBA(image.Rect(0, 0, w, h))
	for y := 0; y < h; y++ {
		for x := 0; x < w; x++ {
			val := d.Input[i*w*h+y*w+x]
			pos := y*img.Stride + x*4
			img.Pix[pos] = 255 - uint8(val*float32(255-col.R))
			img.Pix[pos+1] = 255 - uint8(val*float32(255-col.G))
			img.Pix[pos+2] = 255 - uint8(val*float32(255-col.B))
			img.Pix[pos+3] = 255
		}
	}
	return img
}

// Check if file exists under DataDir
func FileExists(name string) bool {
	filePath := path.Join(DataDir, name)
	_, err := os.Stat(filePath)
	return err == nil
}
