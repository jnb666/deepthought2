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
)

var (
	DataDir   = os.Getenv("GOPATH") + "/src/github.com/jnb666/deepthought2/data"
	DataTypes = []string{"train", "test", "valid"}
)

// Dataset type encapsulates a set of training, test or validation data.
type Dataset struct {
	Data
	Samples       int
	BatchSize     int
	x, xT, y, y1H num.Array
	indexes       []int
	shuffled      bool
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
	d.x = dev.NewArray(num.Float32, d.BatchSize, d.Nfeat())
	d.xT = dev.NewArray(num.Float32, d.Nfeat(), d.BatchSize)
	d.y = dev.NewArray(num.Int32, d.BatchSize)
	d.y1H = dev.NewArray(num.Float32, d.BatchSize, d.Classes)
	return d
}

// Number of batches
func (d *Dataset) Batches() int {
	batches := d.Samples / d.BatchSize
	if d.Samples%d.BatchSize != 0 {
		batches++
	}
	return batches
}

// Get given batch - TODO resize array in case of truncated batch
func (d *Dataset) GetBatch(q num.Queue, b int) (x, y, yOneHot num.Array) {
	nfeat := d.Nfeat()
	start := b * d.BatchSize
	end := start + d.BatchSize
	if end > d.Samples {
		end = d.Samples
	}
	if d.shuffled {
		for i, ix := range d.indexes[start:end] {
			q.Call(
				num.WriteRow(d.x, i, d.Input[ix*nfeat:(ix+1)*nfeat]),
				num.WriteRow(d.y, i, &d.Labels[ix]),
			)
		}
	} else {
		q.Call(
			num.Write(d.xT, d.Input[start*nfeat:end*nfeat]),
			num.Transpose(d.xT, d.x),
			num.Write(d.y, d.Labels[start:end]),
		)
	}
	q.Call(
		num.Onehot(d.y, d.y1H, d.Classes),
	)
	return d.x, d.y, d.y1H
}

// Shuffle the data set
func (d *Dataset) Shuffle() {
	d.indexes = rand.Perm(d.Samples)
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
	fmt.Println("loading data from", name+".dat")
	dec := gob.NewDecoder(f)
	err = dec.Decode(&d)
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
