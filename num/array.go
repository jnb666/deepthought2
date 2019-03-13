package num

import (
	"fmt"
	"unsafe"

	"github.com/jnb666/deepthought2/num/cuda"
	"github.com/jnb666/deepthought2/num/mkl"
)

// Parameters for array printing
var (
	PrintThreshold = 12
	PrintEdgeitems = 4
)

// Buffer interface type represents the underlying data for an array
type Buffer interface {
	// pointer to data
	Data() unsafe.Pointer
	// size of buffer in 32 bit words
	Capacity() int
	// release frees the memory
	Release()
}

func (d cpuDevice) NewBuffer(size int) Buffer {
	return mkl.NewBuffer(size)
}

func (d gpuDevice) NewBuffer(size int) Buffer {
	return cuda.NewBuffer(size)
}

// Allocate a new array using the provided buffer
func NewArray(buf Buffer, dtype DataType, dims ...int) *Array {
	size := Prod(dims)
	if size > buf.Capacity() {
		panic(fmt.Errorf("NewArray: buffer is too small size=%d capacity=%d\n", size, buf.Capacity()))
	}
	return &Array{Buffer: buf, Dtype: dtype, Dims: dims}
}

// Array struct is a general n dimensional tensor similar to a numpy ndarray
// data is stored internally in column major order, may be either on CPU or on GPU depending on buffer type
type Array struct {
	Buffer
	Dtype DataType
	Dims  []int
}

// Size of data in array in words
func (a *Array) Size() int {
	return Prod(a.Dims)
}

// Reshape returns a new array of the same size with a view on the same data but with a different shape
func (a *Array) Reshape(dims ...int) *Array {
	n := a.Size()
	for i := range dims {
		if dims[i] == -1 {
			other := 1
			for j, dim := range dims {
				if i != j {
					if dim == -1 {
						panic("Reshape: can only have single -1 value")
					}
					other *= dim
				}
			}
			dims[i] = n / other
		}
	}
	if Prod(dims) != n {
		panic("reshape must be to array of same size")
	}
	return &Array{Buffer: a.Buffer, Dtype: a.Dtype, Dims: dims}
}

// String returns pretty printed output
func (a *Array) String(q Queue) string {
	var data interface{}
	if a.Dtype == Int32 {
		data = make([]int32, a.Size())
	} else {
		data = make([]float32, a.Size())
	}
	q.Call(Read(a, data)).Finish()
	return format(a.Dims, data, 0, 1, "", false)
}

// array resident in main memory
func (d cpuDevice) NewArray(dtype DataType, dims ...int) *Array {
	return NewArray(mkl.NewBuffer(Prod(dims)), dtype, dims...)
}

func (d cpuDevice) NewArrayLike(a *Array) *Array {
	return d.NewArray(a.Dtype, a.Dims...)
}

// array resident on GPU
func (d gpuDevice) NewArray(dtype DataType, dims ...int) *Array {
	return NewArray(cuda.NewBuffer(Prod(dims)), dtype, dims...)
}

func (d gpuDevice) NewArrayLike(a *Array) *Array {
	return d.NewArray(a.Dtype, a.Dims...)
}

func format(dims []int, data interface{}, at, stride int, indent string, dots bool) string {
	var s string
	switch len(dims) {
	case 0:
		if dots {
			s = "    ... "
		} else {
			switch d := data.(type) {
			case []int32:
				s = fmt.Sprintf("%5d ", d[at])
			case []float32:
				val := d[at]
				if abs(val) < 1 {
					val = float32(int(10000*val+0.5)) / 10000
				}
				s = fmt.Sprintf("%7.5g ", val)
			}
		}
	case 1:
		s = "["
		for i := 0; i < dims[0]; i++ {
			dots2 := dims[0] > PrintThreshold+1 && i == PrintEdgeitems
			s += format([]int{}, data, at+i*stride, 1, "", dots || dots2)
			if dots2 {
				i = dims[0] - PrintEdgeitems - 1
			}
		}
		s += "]"
	case 2:
		var pre, post string
		for i := 0; i < dims[0]; i++ {
			if i == 0 {
				pre = "["
			} else {
				pre = " "
			}
			if i < dims[0]-1 {
				post = "\n"
			} else {
				post = "]\n"
			}
			dots := dims[0] > PrintThreshold+1 && i == PrintEdgeitems
			s += indent + pre + format(dims[1:], data, at+i, dims[0], "", dots) + post
			if dots {
				i = dims[0] - PrintEdgeitems - 1
			}
		}
	default:
		d := len(dims) - 1
		bsize := Prod(dims[:d])
		s = indent + "[\n"
		for i := 0; i < dims[d]; i++ {
			if dims[d] > PrintThreshold+1 && i == PrintEdgeitems {
				s += "   ...  ...   \n"
				i = dims[d] - PrintEdgeitems - 1
			} else {
				s += format(dims[:d], data, at+bsize*i, 1, indent+" ", false)
			}
		}
		s += indent + "]\n"
	}
	return s
}

func abs(x float32) float32 {
	if x >= 0 {
		return x
	}
	return -x
}

// Product of elements of an integer array. Zero dimension array (scalar) has size 1.
func Prod(arr []int) int {
	prod := 1
	for _, v := range arr {
		prod *= v
	}
	return prod
}

// Check if two arrays are the same shape
func SameShape(xd, yd []int) bool {
	if len(xd) != len(yd) {
		return false
	}
	for i := range xd {
		if xd[i] != yd[i] {
			return false
		}
	}
	return true
}

// Total size of one of more arrays in bytes
func Bytes(arr ...*Array) (bytes int) {
	for _, a := range arr {
		if a != nil {
			bytes += 4 * a.Size()
		}
	}
	return bytes
}

// Release one or more arrays or buffers
func Release(arr ...Buffer) {
	for _, a := range arr {
		switch obj := a.(type) {
		case *Array:
			if obj != nil {
				obj.Release()
			}
		case mkl.Buffer:
			obj.Release()
		case cuda.Buffer:
			obj.Release()
		}
	}
}
