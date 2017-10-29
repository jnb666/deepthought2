package img

import (
	"image"
	"image/color"
	"math/rand"
	"strings"
	"testing"
	"time"
)

func char(v uint32) string {
	v = v >> 8
	switch {
	case v == 0:
		return "  "
	case v <= 64:
		return ". "
	case v <= 128:
		return "+ "
	default:
		return "* "
	}
}

func printImage(in image.Image) string {
	r := in.Bounds()
	s := make([]string, r.Dy())
	for y := range s {
		s[y] = "| "
		for x := 0; x < r.Dx(); x++ {
			v, _, _, _ := in.At(x, y).RGBA()
			s[y] += char(v)
		}
		s[y] += "|"
	}
	return strings.Join(s, "\n")
}

func runTest(t *testing.T, mode ConvMode) {
	size := 10
	seed := time.Now().UTC().UnixNano()
	rng := rand.New(rand.NewSource(seed))
	trans := NewTransformer(size, size, mode, rng)

	src := image.NewGray(image.Rect(0, 0, size, size))
	for i := 1; i < size-1; i++ {
		src.Set(size-i-1, i, color.Gray{Y: 255})
	}
	t.Logf("\n%s", printImage(src))

	trans.Trans = Scale
	dst := trans.Transform(src, 0)
	t.Logf("scale\n%s", printImage(dst))

	trans.Trans = Rotate
	dst = trans.Transform(src, 0)
	t.Logf("rotate\n%s", printImage(dst))

	trans.Trans = Elastic
	dst = trans.Transform(src, 0)
	t.Logf("elastic\n%s", printImage(dst))
}

func TestImage(t *testing.T) {
	runTest(t, ConvDefault)
}

func TestAccel(t *testing.T) {
	runTest(t, ConvAccel)
}

func TestBoxBlur(t *testing.T) {
	runTest(t, ConvBoxBlur)
}

var dst image.Image

func runBench(b *testing.B, mode ConvMode, dis TransType) {
	size := 28
	seed := time.Now().UTC().UnixNano()
	rng := rand.New(rand.NewSource(seed))
	data := make([]float32, size*size)
	for i := range data {
		data[i] = rng.Float32()
	}
	src := NewGray(image.Rect(0, 0, size, size), data)
	trans := NewTransformer(size, size, mode, rng)
	trans.Trans = dis
	for i := 0; i < b.N; i++ {
		dst = trans.Transform(src, 0)
	}
}

func BenchmarkAffine(b *testing.B) {
	runBench(b, ConvDefault, Scale+Rotate)
}

func BenchmarkStandard(b *testing.B) {
	runBench(b, ConvDefault, Scale+Rotate+Elastic)
}

func BenchmarkAccel(b *testing.B) {
	runBench(b, ConvAccel, Scale+Rotate+Elastic)
}

func BenchmarkBoxBlur(b *testing.B) {
	runBench(b, ConvBoxBlur, Scale+Rotate+Elastic)
}
