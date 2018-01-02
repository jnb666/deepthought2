package img_test

import (
	"fmt"
	"github.com/jnb666/deepthought2/img"
	"github.com/jnb666/deepthought2/nnet"
	"math/rand"
	"testing"
)

var chars = "  ...+++**"

func printImage(m *img.Image) string {
	s := ""
	w, h := m.Bounds().Dx(), m.Bounds().Dy()
	for channel := 0; channel < 3; channel++ {
		pix := m.Pixels(channel)
		s += fmt.Sprintf("=== channel %d ===\n", channel)
		for y := 0; y < h; y++ {
			for x := 0; x < w; x++ {
				val := int(pix[y+x*h] * 10)
				if val < 0 {
					val = 0
				}
				if val > 9 {
					val = 9
				}
				s += fmt.Sprintf("%c ", chars[val])
			}
			s += "\n"
		}
	}
	return s
}

func TestNorm(t *testing.T) {
	data, err := nnet.LoadDataFile("cifar10_test")
	if err != nil {
		t.Fatal(err)
	}
	d := data.(*img.Data)
	t.Log(printImage(d.Images[3]))

	rng := rand.New(rand.NewSource(42))
	trans := img.NewTransformer(d, img.Normalise, img.ConvAccel, rng)
	m, err := trans.Transform(d.Images[3], 0)
	if err != nil {
		t.Error(err)
	}
	t.Log(printImage(m))
}
