package img

/*
#cgo CFLAGS: -g -O2 -std=c99 -I/opt/intel/mkl/include
#cgo LDFLAGS: -L/opt/intel/mkl/lib/intel64 -L/opt/intel/tbb/lib/intel64/gcc4.7 -Wl,--no-as-needed -lmkl_intel_lp64 -lmkl_tbb_thread -lmkl_core -ltbb -lstdc++ -lpthread -lm -ldl
#include <mkl_vsl.h>
*/
import "C"

import (
	"fmt"
	"math"
	"runtime"
	"unsafe"
)

type ConvMode int

const (
	ConvDefault = iota
	ConvAccel
	ConvBoxBlur
)

func gaussian1d(sigma float64, size int) []float32 {
	kernel := make([]float32, 2*size+1)
	for x := -size; x <= size; x++ {
		d2 := float64(x * x)
		kernel[x+size] = float32(math.Exp(-d2/(2*sigma*sigma)) / (math.Sqrt(2*math.Pi) * sigma))
	}
	return kernel
}

func gaussian2d(sigma float64, size int) []float32 {
	dims := 2*size + 1
	kernel := make([]float32, dims*dims)
	for y := -size; y <= size; y++ {
		for x := -size; x <= size; x++ {
			d2 := float64(x*x + y*y)
			kernel[(y+size)*dims+x+size] = float32(math.Exp(-d2/(2*sigma*sigma)) / (math.Pi * 2 * sigma * sigma))
		}
	}
	return kernel
}

// Convolution to apply kernel to image
type Convolution interface {
	Apply(in, out []float32)
}

// Convolution implemented in go assuming 1d seperable kernel
func NewConv(kernel []float32, ksize, width, height int) Convolution {
	return &conv{w: width, h: height, ksize: ksize, kdata: kernel}
}

type conv struct {
	w, h  int
	ksize int
	kdata []float32
}

func (c *conv) Apply(in, out []float32) {
	copy(out, in)
	c.convH(out, in)
	c.convV(in, out)
}

func (c *conv) convH(in, out []float32) {
	for x := 0; x < c.w; x++ {
		start := max(x-c.ksize, 0)
		end := min(x+c.ksize, c.w-1)
		var sum float32
		for ix := start; ix <= end; ix++ {
			sum += c.kdata[x-ix+c.ksize]
		}
		for y := 0; y < c.h; y++ {
			var val float32
			for ix := start; ix <= end; ix++ {
				val += in[ix+y*c.w] * c.kdata[x-ix+c.ksize]
			}
			out[x+y*c.w] = val / sum
		}
	}
}

func (c *conv) convV(in, out []float32) {
	for y := 0; y < c.h; y++ {
		start := max(y-c.ksize, 0)
		end := min(y+c.ksize, c.h-1)
		var sum float32
		for iy := start; iy <= end; iy++ {
			sum += c.kdata[y-iy+c.ksize]
		}
		for x := 0; x < c.w; x++ {
			var val float32
			for iy := start; iy <= end; iy++ {
				val += in[x+iy*c.w] * c.kdata[y-iy+c.ksize]
			}
			out[x+y*c.w] = val / sum
		}
	}
}

// Gaussian convolution using box blur
func NewConvBox(sigma float64, width, height int) Convolution {
	return &boxConv{w: width, h: height, boxes: boxSize(sigma, 3)}
}

func boxSize(sigma float64, n int) []int {
	wIdeal := math.Sqrt((12 * sigma * sigma / float64(n)) + 1)
	wl := int(wIdeal)
	if wl%2 == 0 {
		wl--
	}
	wu := wl + 2
	mIdeal := (12*sigma*sigma - float64(n*wl*wl-4*n*wl-3*n)) / float64(-4*wl-4)
	m := int(mIdeal + 0.5)
	sizes := make([]int, n)
	for i := range sizes {
		if i < m {
			sizes[i] = wl
		} else {
			sizes[i] = wu
		}
	}
	return sizes
}

type boxConv struct {
	w, h  int
	boxes []int
}

func (c *boxConv) Apply(in, out []float32) {
	for _, size := range c.boxes {
		copy(out, in)
		boxBlurH(out, in, c.w, c.h, (size-1)/2)
		boxBlurV(in, out, c.w, c.h, (size-1)/2)
	}
}

func boxBlurH(in, out []float32, w, h, r int) {
	iarr := 1 / float32(r+r+1)
	for i := 0; i < h; i++ {
		ti := i * w
		li := ti
		ri := ti + r
		fv := in[ti]
		lv := in[ti+w-1]
		val := float32(r+1) * fv
		for j := 0; j < r; j++ {
			val += in[ti+j]
		}
		for j := 0; j <= r; j++ {
			val += in[ri] - fv
			out[ti] = val * iarr
			ri++
			ti++
		}
		for j := r + 1; j < w-r; j++ {
			val += in[ri] - in[li]
			out[ti] = val * iarr
			ri++
			li++
			ti++
		}
		for j := w - r; j < w; j++ {
			val += lv - in[li]
			out[ti] = val * iarr
			li++
			ti++
		}
	}
}

func boxBlurV(in, out []float32, w, h, r int) {
	iarr := 1 / float32(r+r+1)
	for i := 0; i < w; i++ {
		ti := i
		li := ti
		ri := ti + r*w
		fv := in[ti]
		lv := in[ti+w*(h-1)]
		val := float32(r+1) * fv
		for j := 0; j < r; j++ {
			val += in[ti+j*w]
		}
		for j := 0; j <= r; j++ {
			val += in[ri] - fv
			out[ti] = val * iarr
			ri += w
			ti += w
		}
		for j := r + 1; j < h-r; j++ {
			val += in[ri] - in[li]
			out[ti] = val * iarr
			ri += w
			li += w
			ti += w
		}
		for j := h - r; j < h; j++ {
			val += lv - in[li]
			out[ti] = val * iarr
			li += w
			ti += w
		}
	}
}

// Accelerated convolution using Intel MKL libraries
func NewConvMkl(kernel []float32, ksize, width, height int) Convolution {
	c := &mklConv{kdata: kernel}
	dims := shape(width, height)
	kdims := shape(2*ksize+1, 2*ksize+1)
	chk(C.vslsConvNewTaskX(&c.task, C.VSL_CONV_MODE_AUTO, 2, &kdims[0], &dims[0], &dims[0], fptr(c.kdata), nil))
	runtime.SetFinalizer(c, func(obj *mklConv) { obj.Release() })
	return c
}

type mklConv struct {
	task  C.VSLConvTaskPtr
	kdata []float32
}

func (c *mklConv) Apply(in, out []float32) {
	C.vslsConvExecX(c.task, fptr(in), nil, fptr(out), nil)
}

func (c *mklConv) Release() {
	C.vslConvDeleteTask(&c.task)
}

func shape(w, h int) [2]C.int {
	return [2]C.int{C.int(w), C.int(h)}
}

func chk(status C.int) {
	if status != C.VSL_STATUS_OK {
		panic(fmt.Sprintf("MKL VSL error: status %d", status))
	}
}

func fptr(s []float32) *C.float {
	return (*C.float)(unsafe.Pointer(&s[0]))
}

func min(a, b int) int {
	if a < b {
		return a
	}
	return b
}

func max(a, b int) int {
	if a > b {
		return a
	}
	return b
}
