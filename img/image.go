// Package img contains routines for manipulating sets of images.
package img

import (
	"encoding/binary"
	"errors"
	"image"
	"image/color"
)

var (
	RGBModel  = color.ModelFunc(rgbModel)
	GrayModel = color.ModelFunc(grayModel)
)

var ErrDecodeImage = errors.New("error decoding image")

// color is stored as a float for each channel with values in range 0-1
type Color []float32

func (c Color) RGBA() (r, g, b, a uint32) {
	if len(c) == 1 {
		y := clampu(c[0], 0, 1)
		return y, y, y, 0xffff
	}
	return clampu(c[0], 0, 1), clampu(c[1], 0, 1), clampu(c[2], 0, 1), 0xffff
}

func grayModel(c color.Color) color.Color {
	if _, ok := c.(Color); ok {
		return c
	}
	r, g, b, _ := c.RGBA()
	return Color{0.299*float32(r)/0xffff + 0.587*float32(g)/0xffff + 0.114*float32(b)/0xffff}
}

func rgbModel(c color.Color) color.Color {
	if _, ok := c.(Color); ok {
		return c
	}
	r, g, b, _ := c.RGBA()
	return Color{float32(r) / 0xffff, float32(g) / 0xffff, float32(b) / 0xffff}
}

// Image type stores the image data as float32 values in column major order with r, g and b color planes stored separately.
type Image struct {
	Pix      []float32
	Height   int
	Width    int
	Channels int
}

func NewImage(width, height, Channels int) *Image {
	return &Image{Pix: make([]float32, height*width*Channels), Height: height, Width: width, Channels: Channels}
}

func NewImageLike(src *Image) *Image {
	return NewImage(src.Width, src.Height, src.Channels)
}

func (m *Image) MarshalBinary() ([]byte, error) {
	size := m.Height * m.Width * m.Channels
	buf := make([]byte, 3*binary.MaxVarintLen64+size)
	pos := binary.PutVarint(buf, int64(m.Height))
	pos += binary.PutVarint(buf[pos:], int64(m.Width))
	pos += binary.PutVarint(buf[pos:], int64(m.Channels))
	for i := 0; i < size; i++ {
		buf[pos+i] = byte(clamp(m.Pix[i], 0, 1)*255 + 0.5)
	}
	return buf[:pos+size], nil
}

func (m *Image) UnmarshalBinary(data []byte) error {
	var n, pos int
	var val int64
	if val, n = binary.Varint(data[pos:]); n <= 0 {
		return ErrDecodeImage
	}
	pos += n
	m.Height = int(val)
	if val, n = binary.Varint(data[pos:]); n <= 0 {
		return ErrDecodeImage
	}
	pos += n
	m.Width = int(val)
	if val, n = binary.Varint(data[pos:]); n <= 0 {
		return ErrDecodeImage
	}
	pos += n
	m.Channels = int(val)
	size := m.Height * m.Width * m.Channels
	if len(data) != pos+size {
		return ErrDecodeImage
	}
	m.Pix = make([]float32, size)
	for i := 0; i < size; i++ {
		m.Pix[i] = float32(data[pos+i]) / 255
	}
	return nil
}

func (m *Image) TransformType(normalise, distort bool) TransType {
	t := NoTrans
	if normalise {
		t = Normalise
	}
	if distort {
		if m.Channels == 1 {
			t |= GrayTrans
		} else {
			t |= RGBTrans
		}
	}
	return t
}

func (m *Image) ColorModel() color.Model {
	if m.Channels == 1 {
		return GrayModel
	}
	return RGBModel
}

func (m *Image) Bounds() image.Rectangle {
	return image.Rect(0, 0, m.Width, m.Height)
}

func (m *Image) GrayAt(x, y int) float32 {
	if x < 0 || x >= m.Width || y < 0 || y >= m.Height {
		return 0
	}
	return m.Pix[y+x*m.Height]
}

func (m *Image) At(x, y int) color.Color {
	if m.Channels == 1 {
		if x < 0 || x >= m.Width || y < 0 || y >= m.Height {
			return Color{0}
		}
		return Color{m.Pix[y+x*m.Height]}
	} else {
		if x < 0 || x >= m.Width || y < 0 || y >= m.Height {
			return Color{0, 0, 0}
		}
		return Color{
			m.Pix[y+x*m.Height],
			m.Pix[y+x*m.Height+m.Width*m.Height],
			m.Pix[y+x*m.Height+m.Width*m.Height*2],
		}
	}
}

func (m *Image) Set(x, y int, c color.Color) {
	m.SetColor(x, y, m.ColorModel().Convert(c).(Color))
}

func (m *Image) SetColor(x, y int, c Color) {
	if len(c) != m.Channels {
		panic("invalid color format!")
	}
	if x < 0 || x >= m.Width || y < 0 || y >= m.Height {
		return
	}
	for i, val := range c {
		m.Pix[y+x*m.Height+m.Width*m.Height*i] = val
	}
}

func (m *Image) Pixels(ch int) []float32 {
	return m.Pix[ch*m.Width*m.Height : (ch+1)*m.Width*m.Height]
}

// apply highlighting to monochrome image
func Highlight(src *Image, on bool) *Image {
	if src.Channels != 1 {
		return src
	}
	dst := NewImage(src.Width, src.Height, 3)
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

func clampu(x, x0, x1 float32) uint32 {
	return uint32(clamp(x, x0, x1) * 0xffff)
}
