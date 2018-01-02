package img

import (
	"encoding/gob"
	"fmt"
	"github.com/jnb666/deepthought2/stats"
	"io"
)

// Image data set which implements the nnet.Data interface
type Data struct {
	DataHead
	Images []*Image
}

type DataHead struct {
	Epochs int
	Class  []string
	Dims   []int
	Labels []int32
	Mean   []float32
	StdDev []float32
}

// Create a new image set
func NewData(classes []string, labels []int32, images []*Image) *Data {
	src := images[0]
	dims := []int{src.Height, src.Width, src.Channels}
	return &Data{
		DataHead: DataHead{Epochs: 1, Class: classes, Dims: dims, Labels: labels},
		Images:   images,
	}
}

func NewDataLike(d *Data) *Data {
	data := *d
	data.Images = make([]*Image, d.Len())
	return &data
}

// Len function returns number of images
func (d *Data) Len() int { return len(d.Labels) }

// Classes functions number of differerent label values
func (d *Data) Classes() []string { return d.Class }

func (d *Data) ClassSize() int {
	if len(d.Class) > 2 {
		return len(d.Class)
	}
	return 1
}

// Shape returns height, width, channels
func (d *Data) Shape() []int { return d.Dims }

// Label returns classification for given images
func (d *Data) Label(index []int, label []int32) {
	for i, ix := range index {
		label[i] = d.Labels[ix]
	}
}

// Input returns scaled input data in buf array
func (d *Data) Input(index []int, buf []float32, t *Transformer) {
	nfeat := d.nfeat()
	if t == nil {
		for i, ix := range index {
			copy(buf[i*nfeat:], d.Images[ix].Pix)
		}
		return
	}
	temp := t.TransformBatch(index, nil)
	for i := range index {
		copy(buf[i*nfeat:], temp[i].Pix)
	}
}

// Image returns given image number, if channel is set then just show this colour channel
func (d *Data) Image(ix int, channel string) *Image {
	src := d.Images[ix]
	ch, haveChannel := map[string]int{"r": 0, "g": 1, "b": 2}[channel]
	if !haveChannel {
		return src
	}
	dst := NewImageLike(src)
	for i := 0; i < src.Channels; i++ {
		copy(dst.Pixels(i), src.Pixels(ch))
	}
	return dst
}

// Slice returns images from start to end
func (d *Data) Slice(start, end int) *Data {
	data := *d
	data.Labels = append([]int32{}, d.Labels[start:end]...)
	data.Images = append([]*Image{}, d.Images[start:end]...)
	return &data
}

func (d *Data) nfeat() int {
	n := 1
	for _, d := range d.Dims {
		n *= d
	}
	return n
}

// Encode data to binary file
func (d *Data) Encode(w io.Writer) error {
	enc := gob.NewEncoder(w)
	if err := enc.Encode(&d.DataHead); err != nil {
		return fmt.Errorf("error encoding header: %s", err)
	}
	for i, img := range d.Images {
		if err := enc.Encode(img); err != nil {
			return fmt.Errorf("error encoding image %d: %s", i, err)
		}
	}
	return nil
}

// Decode data from binary file
func (d *Data) Decode(r io.Reader) error {
	d.DataHead = DataHead{}
	dec := gob.NewDecoder(r)
	if err := dec.Decode(&d.DataHead); err != nil {
		return fmt.Errorf("error decoding header: %s", err)
	}
	d.Images = make([]*Image, d.Len())
	for i := range d.Images {
		if err := dec.Decode(&d.Images[i]); err != nil {
			return fmt.Errorf("error decoding image %d: %s", i, err)
		}
	}
	return nil
}

// Calculate mean and stddev from set of images
func GetStats(imgList ...[]*Image) (mean, std []float32) {
	channels := imgList[0][0].Channels
	stat := make([]*stats.Average, channels)
	for i := range stat {
		stat[i] = new(stats.Average)
	}
	for _, images := range imgList {
		for _, img := range images {
			for ch, s := range stat {
				for _, val := range img.Pixels(ch) {
					s.Add(float64(val))
				}
			}
		}
	}
	mean = make([]float32, channels)
	std = make([]float32, channels)
	for i, s := range stat {
		mean[i] = float32(s.Mean)
		std[i] = float32(s.StdDev)
	}
	fmt.Printf("mean = %.2f stddev = %.2f\n", mean, std)
	return mean, std
}
