package img

import (
	"fmt"
	"image"
	"image/color"
	"math/rand"
	"strings"
	"testing"
	"time"
)

func printArray(in []float32, size int) string {
	s := make([]string, size)
	for i := 0; i < size; i++ {
		s[i] = fmt.Sprintf("%6.3f", in[i*size:(i+1)*size])
	}
	return strings.Join(s, "\n")
}

func runTest(t *testing.T, accel bool) {
	seed := time.Now().UTC().UnixNano()
	rng := rand.New(rand.NewSource(seed))
	trans := NewTransformer(8, 8, 1, rng, accel)

	src := image.NewGray(image.Rect(0, 0, 8, 8))
	for i := 1; i < 7; i++ {
		src.Set(7-i, i, color.Gray{Y: 255})
	}
	data := make([]float32, 64)
	Unpack(src, data)
	t.Logf("\n%s", printArray(data, 8))

	trans.Transform(src, Scale, data, 0)
	t.Logf("scale\n%s", printArray(data, 8))

	trans.Transform(src, Rotate, data, 0)
	t.Logf("rotate\n%s", printArray(data, 8))

	trans.Transform(src, Elastic, data, 0)
	t.Logf("elastic\n%s", printArray(data, 8))
}

func TestImage(t *testing.T) {
	runTest(t, false)
}

func TestAccel(t *testing.T) {
	runTest(t, true)
}

func runBench(b *testing.B, accel bool) {
	seed := time.Now().UTC().UnixNano()
	rng := rand.New(rand.NewSource(seed))
	data := make([]float32, 28*28)
	for i := range data {
		data[i] = rng.Float32()
	}
	src := NewGray(image.Rect(0, 0, 28, 28), data)
	trans := NewTransformer(28, 28, 1, rng, accel)
	for i := 0; i < b.N; i++ {
		trans.Transform(src, Scale+Rotate+Elastic, data, 0)
	}
}

func BenchmarkImage(b *testing.B) {
	runBench(b, false)
}

func BenchmarkAccel(b *testing.B) {
	runBench(b, true)
}
