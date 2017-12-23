// Package img contains routines for manipulating sets of images.
package img

import (
	"encoding/gob"
	"fmt"
	"image"
	"io"
	"math"
	"math/rand"
	"runtime"
	"sync"
)

// Types of image transformations
type TransType int

const (
	Scale TransType = 1 << iota
	Rotate
	Elastic
)

var (
	MaxScale     = 0.15
	MaxRotate    = 15.0
	ElasticScale = 0.5
	KernelSize   = 9
	KernelSigma  = 4.0
)

var gaussian1, gaussian2 []float32

// Image data set which implements the nnet.Data interface
type Data struct {
	Epochs int
	Class  []string
	Dims   []int
	Labels []int32
	Images []image.Image
	file   io.ReadSeeker
}

func init() {
	gob.Register(&Data{})
	gob.Register(&image.Gray{})
	gob.Register(&image.NRGBA{})
	gaussian1 = gaussian1d(KernelSigma, KernelSize)
	gaussian2 = gaussian2d(KernelSigma, KernelSize)
}

// Create a new image set
func NewData(classes []string, labels []int32, images []image.Image) *Data {
	src := images[0]
	b := src.Bounds()
	dims := []int{b.Dy(), b.Dx(), 0}
	switch src.(type) {
	case *image.Gray:
		dims[2] = 1
	case *image.NRGBA:
		dims[2] = 3
	default:
		panic(fmt.Sprintf("NewData: image type %T not supported", src))
	}
	return &Data{Epochs: 1, Class: classes, Dims: dims, Labels: labels, Images: images}
}

// Len function returns number of images
func (d *Data) Len() int { return len(d.Labels) }

// Classes functions number of differerent label values
func (d *Data) Classes() []string { return d.Class }

// Shape returns height, width
func (d *Data) Shape() []int { return d.Dims }

// Image returns given image number
func (d *Data) Image(ix int) image.Image { return d.Images[ix] }

// Label returns classification for given images
func (d *Data) Label(index []int, label []int32) {
	for i, ix := range index {
		label[i] = d.Labels[ix]
	}
}

// Input returns scaled input data in buf array
func (d *Data) Input(index []int, buf []float32) {
	nfeat := 1
	for _, d := range d.Dims {
		nfeat *= d
	}
	for i, ix := range index {
		Unpack(d.Images[ix], buf[i*nfeat:(i+1)*nfeat])
	}
}

// Slice returns images from start to end
func (d *Data) Slice(start, end int) *Data {
	return &Data{
		Epochs: d.Epochs,
		Class:  d.Class,
		Dims:   d.Dims,
		Labels: d.Labels[start:end],
		Images: d.Images[start:end],
	}
}

func (d *Data) SetFile(f io.ReadSeeker) { d.file = f }

func (d *Data) File() io.ReadSeeker {
	if d.Epochs > 1 {
		return d.file
	}
	return nil
}

// Create a new grayscale image from data buffer
func NewGray(r image.Rectangle, data []float32) *image.Gray {
	dst := image.NewGray(r)
	for i, val := range data {
		dst.Pix[i] = toInt8(val)
	}
	return dst
}

// Unpack data buffer from image
func Unpack(in image.Image, data []float32) {
	switch src := in.(type) {
	case *image.Gray:
		for j, pix := range src.Pix {
			data[j] = float32(pix) / 255
		}
	case *image.NRGBA:
		npix := len(src.Pix) / 4
		for j := 0; j < npix; j++ {
			data[j] = float32(src.Pix[j*4]) / 255
			data[j+npix] = float32(src.Pix[j*4+1]) / 255
			data[j+2*npix] = float32(src.Pix[j*4+2]) / 255
		}
	default:
		panic(fmt.Sprintf("Unpack: image type %T not supported", src))
	}
}

// apply highlighting to monochrome image
func Highlight(in image.Image, on bool) image.Image {
	switch src := in.(type) {
	case *image.Gray:
		dst := image.NewNRGBA(src.Bounds())
		for j, pix := range src.Pix {
			if on {
				dst.Pix[j*4] = 255
			} else {
				dst.Pix[j*4] = 255 - pix
			}
			dst.Pix[j*4+1] = 255 - pix
			dst.Pix[j*4+2] = 255 - pix
			dst.Pix[j*4+3] = 255
		}
		return dst
	default:
		return in
	}
}

type Transformer struct {
	Amount float64
	Trans  TransType
	w, h   int
	rng    []*rand.Rand
	conv   Convolution
}

// Create a new transformer object which applies a sequency of image transformations
func NewTransformer(w, h int, mode ConvMode, rng *rand.Rand) *Transformer {
	threads := runtime.GOMAXPROCS(0)
	t := &Transformer{Amount: 1, Trans: Scale + Rotate + Elastic, w: w, h: h, rng: make([]*rand.Rand, threads)}
	for i := range t.rng {
		t.rng[i] = rand.New(rand.NewSource(rng.Int63()))
	}
	switch mode {
	case ConvDefault:
		t.conv = NewConv(gaussian1, KernelSize, w, h)
	case ConvAccel:
		t.conv = NewConvMkl(gaussian2, KernelSize, w, h)
	case ConvBoxBlur:
		t.conv = NewConvBox(KernelSigma, w, h)
	default:
		panic("invalid convolution mode")
	}
	return t
}

// Transform a batch of images in parallel
func (t *Transformer) TransformBatch(index []int, src, dst []image.Image) {
	var wg sync.WaitGroup
	queue := make(chan int, len(t.rng))
	for thread := range t.rng {
		wg.Add(1)
		go func(thread int) {
			for i := range queue {
				ix := index[i]
				dst[i] = t.Transform(src[ix], thread)
			}
			wg.Done()
		}(thread)
	}
	for i := range index {
		queue <- i
	}
	close(queue)
	wg.Wait()
}

// Generate a scaling, rotation or elastic image transformation
func (t *Transformer) Transform(src image.Image, thread int) image.Image {
	rng := t.rng[thread]
	dx := make([]float32, t.w*t.h)
	dy := make([]float32, t.w*t.h)
	var elX, elY float32
	if t.Trans&Elastic != 0 {
		ux := make([]float32, t.w*t.h)
		uy := make([]float32, t.w*t.h)
		for i := range ux {
			ux[i] = rng.Float32()*2 - 1
			uy[i] = rng.Float32()*2 - 1
		}
		t.conv.Apply(ux, dx)
		t.conv.Apply(uy, dy)
		elX = float32(t.Amount*ElasticScale) * float32(t.w)
		elY = float32(t.Amount*ElasticScale) * float32(t.h)
	}
	var sx, sy float32
	if t.Trans&Scale != 0 {
		sx = float32(t.Amount*MaxScale) * (2*rng.Float32() - 1)
		sy = float32(t.Amount*MaxScale) * (2*rng.Float32() - 1)
	}
	var sina, cosa float32
	if t.Trans&Rotate != 0 {
		angle := t.Amount * MaxRotate * (math.Pi / 180) * (2*rng.Float64() - 1)
		sa, ca := math.Sincos(angle)
		sina, cosa = float32(sa), float32(ca-1)
	}
	for y := 0; y < t.h; y++ {
		ym := float32(2*y-t.h+1) / 2
		for x := 0; x < t.w; x++ {
			xm := float32(2*x-t.w+1) / 2
			dx[x+y*t.w] = dx[x+y*t.w]*elX + xm*(sx+cosa) - ym*sina
			dy[x+y*t.w] = dy[x+y*t.w]*elY + ym*(sy+cosa) + xm*sina
		}
	}
	return t.Sample(src, dx, dy)
}

// Apply the transform to the image and interpolate the results
func (t *Transformer) Sample(src image.Image, dx, dy []float32) image.Image {
	switch in := src.(type) {
	case *image.Gray:
		out := image.NewGray(src.Bounds())
		pixel := func(ix, iy int) float32 {
			if ix < 0 || ix >= t.w || iy < 0 || iy >= t.h {
				return 0
			}
			return float32(in.Pix[iy*t.w+ix]) / 255
		}
		for y := 0; y < t.h; y++ {
			for x := 0; x < t.w; x++ {
				pos := x + y*t.w
				xv := float32(x) + dx[pos]
				yv := float32(y) + dy[pos]
				ix, iy := int(xv), int(yv)
				xf, yf := xv-float32(ix), yv-float32(iy)
				avg0 := pixel(ix, iy)*(1-xf) + pixel(ix+1, iy)*xf
				avg1 := pixel(ix, iy+1)*(1-xf) + pixel(ix+1, iy+1)*xf
				out.Pix[y*t.w+x] = toInt8(avg0*(1-yf) + avg1*yf)
			}
		}
		return out
	default:
		panic(fmt.Sprintf("image type %T not supported", src))
	}
}

func toInt8(x float32) uint8 {
	if x <= 0 {
		return 0
	}
	if x >= 1 {
		return 255
	}
	return uint8(255 * x)
}
