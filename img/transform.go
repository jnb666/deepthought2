package img

import (
	"fmt"
	"math"
	"math/rand"
	"runtime"
	"sort"
	"strings"
	"sync"
)

const epsilon = 1e-5

// Types of image transformations
type TransType int

const NoTrans TransType = 0

const (
	Scale TransType = 1 << iota
	Rotate
	Elastic
	HorizFlip
	Pan
	Normalise
)

var (
	GrayTrans = Scale | Rotate | Elastic
	RGBTrans  = HorizFlip | Pan
)

var transTypeNames = map[TransType]string{
	Scale:     "Scale",
	Rotate:    "Rotate",
	Elastic:   "Elastic",
	HorizFlip: "HorizFlip",
	Pan:       "Pan",
	Normalise: "Normalise",
}

func (t TransType) String() string {
	if t == NoTrans {
		return "None"
	}
	s := []string{}
	for key, name := range transTypeNames {
		if t&key != 0 {
			s = append(s, name)
		}
	}
	sort.Strings(s)
	return strings.Join(s, " ")
}

var (
	MaxScale     = 0.15
	MaxRotate    = 15.0
	ElasticScale = 0.5
	KernelSize   = 9
	KernelSigma  = 4.0
	PanPixels    = 4
)

type Transformer struct {
	Amount float64
	Trans  TransType
	data   *Data
	w, h   int
	rng    []*rand.Rand
	conv   Convolution
}

// Create a new transformer object which applies a sequency of image transformations
func NewTransformer(data *Data, trans TransType, mode ConvMode, rng *rand.Rand) *Transformer {
	threads := runtime.GOMAXPROCS(0)
	b := data.Images[0].Bounds()
	t := &Transformer{Amount: 1, Trans: trans, data: data, w: b.Dx(), h: b.Dy()}
	for i := 0; i < threads; i++ {
		t.rng = append(t.rng, rand.New(rand.NewSource(rng.Int63())))
	}
	switch mode {
	case ConvDefault:
		t.conv = NewConv(gaussian1, KernelSize, t.w, t.h)
	case ConvAccel:
		t.conv = NewConvMkl(gaussian2, KernelSize, t.w, t.h)
	case ConvBoxBlur:
		t.conv = NewConvBox(KernelSigma, t.w, t.h)
	default:
		panic("invalid convolution mode")
	}
	return t
}

// Transform a batch of images in parallel
func (t *Transformer) TransformBatch(index []int, dst []Image) []Image {
	if dst == nil {
		dst = make([]Image, len(index))
	}
	var wg sync.WaitGroup
	queue := make(chan int, len(t.rng))
	for thread := range t.rng {
		wg.Add(1)
		go func(thread int) {
			var err error
			for i := range queue {
				ix := index[i]
				dst[i], err = t.Transform(t.data.Images[ix], thread)
				if err != nil {
					panic(err)
				}
			}
			wg.Done()
		}(thread)
	}
	for i := range index {
		queue <- i
	}
	close(queue)
	wg.Wait()
	return dst
}

// Perform one or more image transforms
func (t *Transformer) Transform(img Image, thread int) (Image, error) {
	rng := t.rng[thread]
	if t.Trans&(Scale|Rotate|Elastic) != 0 {
		if m, ok := img.(*GrayImage); ok {
			img = t.transformGray(m, thread)
		} else {
			return img, fmt.Errorf("ImageTransformer: image type %T not supported for %s", img, t.Trans)
		}
	}
	if t.Trans&HorizFlip != 0 && rng.Float64() > 0.5 {
		img = transform(img, func(x, y int) (int, int) { return t.w - x - 1, y })
	}
	if t.Trans&Pan != 0 {
		off := int(float64(PanPixels)*t.Amount + 0.5)
		ox := rng.Intn(2*off+1) - PanPixels
		oy := rng.Intn(2*off+1) - PanPixels
		if ox != 0 || oy != 0 {
			img = transform(img, func(x, y int) (int, int) { return wrap(x-ox, t.w), wrap(y-oy, t.h) })
		}
	}
	var err error
	if t.Trans&Normalise != 0 {
		img, err = t.normalise(img)
	}
	return img, err
}

func (t *Transformer) normalise(src Image) (Image, error) {
	channels := src.Channels()
	if t.data.Mean == nil || t.data.StdDev == nil || len(t.data.Mean) != channels || len(t.data.StdDev) != channels {
		return src, fmt.Errorf("error applying normalisation - missing mean and stddev")
	}
	dst := NewImageLike(src)
	for ch := 0; ch < channels; ch++ {
		pix := dst.Pixels(ch)
		for i, val := range src.Pixels(ch) {
			pix[i] = (val - t.data.Mean[ch]) / t.data.StdDev[ch]
		}
	}
	return dst, nil
}

func (t *Transformer) transformGray(src *GrayImage, thread int) *GrayImage {
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
	dst := NewGray(t.w, t.h)
	for y := 0; y < t.h; y++ {
		for x := 0; x < t.w; x++ {
			pos := x + y*t.w
			xv := float32(x) + dx[pos]
			yv := float32(y) + dy[pos]
			ix, iy := int(xv), int(yv)
			xf, yf := xv-float32(ix), yv-float32(iy)
			avg0 := src.GrayAt(ix, iy).Y*(1-xf) + src.GrayAt(ix+1, iy).Y*xf
			avg1 := src.GrayAt(ix, iy+1).Y*(1-xf) + src.GrayAt(ix+1, iy+1).Y*xf
			dst.Set(x, y, Gray{Y: avg0*(1-yf) + avg1*yf})
		}
	}
	return dst
}

func transform(src Image, fn func(x, y int) (int, int)) Image {
	dst := NewImageLike(src)
	dx, dy := src.Bounds().Dx(), src.Bounds().Dy()
	for y := 0; y < dy; y++ {
		for x := 0; x < dx; x++ {
			sx, sy := fn(x, y)
			dst.Set(x, y, src.At(sx, sy))
		}
	}
	return dst
}

func wrap(x, dx int) int {
	if x < 0 {
		return -x - 1
	}
	if x >= dx {
		return 2*dx - x - 1
	}
	return x
}

func clamp(x, x0, x1 float32) float32 {
	if x < x0 {
		return x0
	}
	if x > x1 {
		return x1
	}
	return x
}
