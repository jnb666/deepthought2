// Package img contains routines for manipulating sets of images.
package img

import (
	"encoding/gob"
	"fmt"
	"image"
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
	NClass int
	Dims   []int
	Labels []int32
	Images []image.Image
	trans  *Transformer
}

func init() {
	gob.Register(&Data{})
	gob.Register(&image.Gray{})
	gaussian1 = gaussian1d(KernelSigma, KernelSize)
	gaussian2 = gaussian2d(KernelSigma, KernelSize)
}

// Create a new image set
func NewData(classes int, labels []int32, images []image.Image) *Data {
	b := images[0].Bounds()
	dims := []int{b.Dy(), b.Dx()}
	return &Data{NClass: classes, Dims: dims, Labels: labels, Images: images}
}

// Len function returns number of images
func (d *Data) Len() int { return len(d.Labels) }

// Classes functions number of differerent label values
func (d *Data) Classes() int { return d.NClass }

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

// Input returns scaled input data in buf array, optionally applying transform if set
func (d *Data) Input(index []int, buf []float32) {
	w, h := d.Dims[1], d.Dims[0]
	nfeat := w * h
	if d.trans == nil {
		for i, ix := range index {
			Unpack(d.Images[ix], buf[i*nfeat:(i+1)*nfeat])
		}
		return
	}
	var wg sync.WaitGroup
	queue := make(chan int, len(d.trans.rng))
	for thread := range d.trans.rng {
		wg.Add(1)
		go func(thread int) {
			for i := range queue {
				ix := index[i]
				d.trans.Transform(d.Images[ix], Scale+Rotate+Elastic, buf[i*nfeat:(i+1)*nfeat], thread)
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

// Setup new image transformer to apply distortion
func (d *Data) Distort(amount float64, rng *rand.Rand, accel bool) {
	dims := d.Shape()
	d.trans = NewTransformer(dims[1], dims[0], amount, rng, accel)
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
	default:
		panic(fmt.Sprintf("image type %T not supported", src))
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
	w, h   int
	rng    []*rand.Rand
	conv   Convolution
}

// Create a new transformer object, if accel is set then use Intel MKL convolution libraries
func NewTransformer(w, h int, intensity float64, rng *rand.Rand, accel bool) *Transformer {
	threads := runtime.GOMAXPROCS(0)
	t := &Transformer{Amount: intensity, w: w, h: h, rng: make([]*rand.Rand, threads)}
	for i := range t.rng {
		t.rng[i] = rand.New(rand.NewSource(rng.Int63()))
	}
	if accel {
		t.conv = NewConvMkl(gaussian2, KernelSize, w, h)
	} else {
		t.conv = NewConv(gaussian1, KernelSize, w, h)
	}
	return t
}

// Generate a scaling, rotation or elastic image transformation
func (t *Transformer) Transform(src image.Image, trans TransType, data []float32, thread int) {
	rng := t.rng[thread]
	dx := make([]float32, t.w*t.h)
	dy := make([]float32, t.w*t.h)
	var elX, elY float32
	if trans&Elastic != 0 {
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
	if trans&Scale != 0 {
		sx = float32(t.Amount*MaxScale) * (2*rng.Float32() - 1)
		sy = float32(t.Amount*MaxScale) * (2*rng.Float32() - 1)
	}
	var sina, cosa float32
	if trans&Rotate != 0 {
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
	t.sample(src, dx, dy, data)
}

// Apply the transform to the image and interpolate the results
func (t *Transformer) sample(in image.Image, dx, dy, data []float32) {
	switch src := in.(type) {
	case *image.Gray:
		pixel := func(ix, iy int) float32 {
			if ix < 0 || ix >= t.w || iy < 0 || iy >= t.h {
				return 0
			}
			return float32(src.Pix[iy*t.w+ix]) / 255
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
				data[y*t.w+x] = clampf(avg0*(1-yf) + avg1*yf)
			}
		}

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

func clampf(x float32) float32 {
	if x <= 0 {
		return 0
	}
	if x >= 1 {
		return 1
	}
	return x
}
