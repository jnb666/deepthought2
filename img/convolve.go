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
type conv struct {
	w, h  int
	ksize int
	kdata []float32
}

func NewConv(kernel []float32, ksize, width, height int) Convolution {
	return &conv{w: width, h: height, ksize: ksize, kdata: kernel}
}

func (c *conv) Apply(in, out []float32) {
	temp := make([]float32, c.w*c.h)
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
			temp[x+y*c.w] = val / sum
		}
	}
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
				val += temp[x+iy*c.w] * c.kdata[y-iy+c.ksize]
			}
			out[x+y*c.w] = val / sum
		}
	}
}

// Accelerated convolution using Intel MKL libraries
type mklConv struct {
	task  C.VSLConvTaskPtr
	kdata []float32
}

func NewConvMkl(kernel []float32, ksize, width, height int) Convolution {
	c := &mklConv{kdata: kernel}
	dims := shape(width, height)
	kdims := shape(2*ksize+1, 2*ksize+1)
	chk(C.vslsConvNewTaskX(&c.task, C.VSL_CONV_MODE_AUTO, 2, &kdims[0], &dims[0], &dims[0], fptr(c.kdata), nil))
	runtime.SetFinalizer(c, func(obj *mklConv) { obj.Release() })
	return c
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
