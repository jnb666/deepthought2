// Package img contains routines for manipulating sets of images.
package img

import (
	"image"
	"image/color"
	"image/draw"
)

var (
	GrayModel = color.ModelFunc(grayModel)
	RGBModel  = color.ModelFunc(rgbModel)
)

// Gray color stored a float in range 0-1
type Gray struct {
	Y float32
}

func (c Gray) RGBA() (r, g, b, a uint32) {
	y := clampu(c.Y, 0, 1)
	return y, y, y, 0xffff
}

func grayModel(c color.Color) color.Color {
	if _, ok := c.(Gray); ok {
		return c
	}
	r, g, b, _ := c.RGBA()
	return Gray{Y: 0.299*float32(r)/0xffff + 0.587*float32(g)/0xffff + 0.114*float32(b)/0xffff}
}

// RGB color is stored as a float for each channel with values in range 0-1
type RGB struct {
	R, G, B float32
}

func (c RGB) RGBA() (r, g, b, a uint32) {
	return clampu(c.R, 0, 1), clampu(c.G, 0, 1), clampu(c.B, 0, 1), 0xffff
}

func rgbModel(c color.Color) color.Color {
	if _, ok := c.(RGB); ok {
		return c
	}
	r, g, b, _ := c.RGBA()
	return RGB{R: float32(r) / 0xffff, G: float32(g) / 0xffff, B: float32(b) / 0xffff}
}

// Image interface type with additional method to get the pixel data
type Image interface {
	draw.Image
	Pixels(ch int) []float32
	Channels() int
	TransformType(normalise, distort bool) TransType
}

func NewImageLike(src Image) Image {
	switch m := src.(type) {
	case *GrayImage:
		return NewGray(m.Width, m.Height)
	case *RGBImage:
		return NewRGB(m.Width, m.Height)
	default:
		panic("invalid image type")
	}
}

// GrayImage type stores the image data as float32 values in column major order.
type GrayImage struct {
	Pix    []float32
	Height int
	Width  int
}

func NewGray(width, height int) *GrayImage {
	return &GrayImage{Pix: make([]float32, height*width), Height: height, Width: width}
}

func (m *GrayImage) Channels() int {
	return 1
}

func (m *GrayImage) TransformType(normalise, distort bool) TransType {
	t := NoTrans
	if normalise {
		t = Normalise
	}
	if distort {
		t |= GrayTrans
	}
	return t
}

func (m *GrayImage) ColorModel() color.Model {
	return GrayModel
}

func (m *GrayImage) Bounds() image.Rectangle {
	return image.Rect(0, 0, m.Width, m.Height)
}

func (m *GrayImage) GrayAt(x, y int) Gray {
	if x < 0 || x >= m.Width || y < 0 || y >= m.Height {
		return Gray{}
	}
	return Gray{Y: m.Pix[y+x*m.Height]}
}

func (m *GrayImage) At(x, y int) color.Color {
	return m.GrayAt(x, y)
}

func (m *GrayImage) Set(x, y int, c color.Color) {
	if x < 0 || x >= m.Width || y < 0 || y >= m.Height {
		return
	}
	m.Pix[y+x*m.Height] = grayModel(c).(Gray).Y
}

func (m *GrayImage) Pixels(ch int) []float32 {
	return m.Pix
}

// RGBImage type stores the image data as float32 values in column major order with r, g and b color planes stored separately.
type RGBImage struct {
	Pix    []float32
	Height int
	Width  int
}

func NewRGB(width, height int) *RGBImage {
	return &RGBImage{Pix: make([]float32, height*width*3), Height: height, Width: width}
}

func (m *RGBImage) Channels() int {
	return 3
}

func (m *RGBImage) TransformType(normalise, distort bool) TransType {
	t := NoTrans
	if normalise {
		t = Normalise
	}
	if distort {
		t |= RGBTrans
	}
	return t
}

func (m *RGBImage) ColorModel() color.Model {
	return RGBModel
}

func (m *RGBImage) Bounds() image.Rectangle {
	return image.Rect(0, 0, m.Width, m.Height)
}

func (m *RGBImage) RGBAt(x, y int) RGB {
	if x < 0 || x >= m.Width || y < 0 || y >= m.Height {
		return RGB{}
	}
	r := m.Pix[y+x*m.Height]
	g := m.Pix[y+x*m.Height+m.Width*m.Height]
	b := m.Pix[y+x*m.Height+2*m.Width*m.Height]
	return RGB{R: r, G: g, B: b}
}

func (m *RGBImage) At(x, y int) color.Color {
	return m.RGBAt(x, y)
}

func (m *RGBImage) Set(x, y int, c color.Color) {
	if x < 0 || x >= m.Width || y < 0 || y >= m.Height {
		return
	}
	rgb := rgbModel(c).(RGB)
	m.Pix[y+x*m.Height] = rgb.R
	m.Pix[y+x*m.Height+m.Width*m.Height] = rgb.G
	m.Pix[y+x*m.Height+2*m.Width*m.Height] = rgb.B
}

func (m *RGBImage) Pixels(ch int) []float32 {
	if ch >= 0 && ch <= 2 {
		return m.Pix[ch*m.Width*m.Height : (ch+1)*m.Width*m.Height]
	}
	return m.Pix
}

// apply highlighting to monochrome image
func Highlight(in Image, on bool) Image {
	if src, ok := in.(*GrayImage); ok {
		dst := NewRGB(src.Width, src.Height)
		for ch := 0; ch < 3; ch++ {
			for j, pix := range src.Pix {
				val := 1 - pix
				if on && ch == 0 {
					val = 1
				}
				dst.Pix[ch*src.Width*src.Height+j] = val
			}
		}
		return dst
	}
	return in
}

func clampu(x, x0, x1 float32) uint32 {
	return uint32(clamp(x, x0, x1) * 0xffff)
}
