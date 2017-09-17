package num

import (
	"fmt"
	"github.com/jnb666/deepthought2/num/mkl"
	"unsafe"
)

// Parameters for array printing
var (
	PrintThreshold = 12
	PrintEdgeitems = 4
)

// Array interface is a general n dimensional tensor similar to a numpy ndarray
// data is stored internally in column major order
type Array interface {
	// Dims returns the shape of the array in rows, cols, ... order
	Dims() []int
	// Dtype returns the data type of the elements in the array
	Dtype() DataType
	// Reshape returns a new array of the same size with a view on the same data but with a different shape
	Reshape(dims ...int) Array
	// Reference to the raw data
	Data() unsafe.Pointer
	// Formatted output
	String(q Queue) string
}

// Allocate a new array on the device with the given type and fill with zeroes.
func (d cpuDevice) NewArray(dtype DataType, dims ...int) Array {
	return &arrayCPU{
		dims:  dims,
		dtype: dtype,
		data:  mkl.NewBuffer(Prod(dims)),
	}
}

// Allocate a new array with same device, datatype and shape as the input.
func (d cpuDevice) NewArrayLike(a Array) Array {
	return d.NewArray(a.Dtype(), a.Dims()...)
}

type arrayCPU struct {
	dims  []int
	dtype DataType
	data  unsafe.Pointer
}

func (a *arrayCPU) Dims() []int { return a.dims }

func (a *arrayCPU) Dtype() DataType { return a.dtype }

func (a *arrayCPU) Data() unsafe.Pointer { return a.data }

func (a *arrayCPU) Reshape(dims ...int) Array {
	n := Prod(a.Dims())
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
	if Prod(a.dims) != Prod(dims) {
		panic("reshape must be to array of same size")
	}
	return &arrayCPU{dims: dims, dtype: a.dtype, data: a.data}
}

func (a *arrayCPU) String(q Queue) string {
	var data interface{}
	n := Prod(a.Dims())
	if a.Dtype() == Int32 {
		data = make([]int32, n)
	} else {
		data = make([]float32, n)
	}
	q.Call(Read(a, data)).Finish()
	return format(a.Dims(), data, 0, 1, "", false)
}

func abs(x float32) float32 {
	if x >= 0 {
		return x
	}
	return -x
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
		//fmt.Printf("format2 %v %d %d %v\n", dims, at, stride, dots)
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
		//fmt.Printf("formatn %v %d %d %v\n", dims, at, stride, dots)
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
