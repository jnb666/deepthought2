package num

import (
	"fmt"
	"github.com/jnb666/deepthought2/num/cuda"
	"github.com/jnb666/deepthought2/num/mkl"
	"runtime"
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
	// Size is total number of elements
	Size() int
	// Dtype returns the data type of the elements in the array
	Dtype() DataType
	// Reshape returns a new array of the same size with a view on the same data but with a different shape
	Reshape(dims ...int) Array
	// Reference to the raw data
	Data() unsafe.Pointer
	// Formatted output
	String(q Queue) string
	// Release any allocated memory
	Release()
}

// array resident in main memory
type arrayCPU struct {
	arrayBase
	data unsafe.Pointer
}

func (d cpuDevice) NewArray(dtype DataType, dims ...int) Array {
	return newArrayCPU(dtype, dims, mkl.NewBuffer(Prod(dims)))
}

func (d cpuDevice) NewArrayLike(a Array) Array {
	return newArrayCPU(a.Dtype(), a.Dims(), mkl.NewBuffer(Prod(a.Dims())))
}

func newArrayCPU(dtype DataType, dims []int, data unsafe.Pointer) *arrayCPU {
	return &arrayCPU{arrayBase: arrayBase{size: Prod(dims), dims: dims, dtype: dtype}, data: data}
}

func (a *arrayCPU) Data() unsafe.Pointer { return a.data }

func (a *arrayCPU) Release() {}

func (a *arrayCPU) Reshape(dims ...int) Array {
	return &arrayCPU{arrayBase: a.reshape(dims), data: a.data}
}

func (a *arrayCPU) String(q Queue) string { return toString(a, q) }

// array resident on GPU
type arrayGPU struct {
	arrayBase
	data cuda.Buffer
}

func (d gpuDevice) NewArray(dtype DataType, dims ...int) Array {
	return newArrayGPU(dtype, dims, cuda.NewBuffer(Prod(dims)*4).Clear())
}

func (d gpuDevice) NewArrayLike(a Array) Array {
	return newArrayGPU(a.Dtype(), a.Dims(), cuda.NewBuffer(Prod(a.Dims())*4).Clear())
}

func newArrayGPU(dtype DataType, dims []int, data cuda.Buffer) *arrayGPU {
	a := &arrayGPU{arrayBase: arrayBase{size: Prod(dims), dims: dims, dtype: dtype}, data: data}
	runtime.SetFinalizer(a, func(obj *arrayGPU) { a.data.Free() })
	return a
}

func (a *arrayGPU) Data() unsafe.Pointer { return a.data.Ptr }

func (a *arrayGPU) Release() { a.data.Free() }

func (a *arrayGPU) Reshape(dims ...int) Array {
	return &arrayGPU{arrayBase: a.reshape(dims), data: a.data}
}

func (a *arrayGPU) String(q Queue) string { return toString(a, q) }

// common array functions
type arrayBase struct {
	size  int
	dims  []int
	dtype DataType
}

func (a arrayBase) Size() int { return a.size }

func (a arrayBase) Dims() []int { return a.dims }

func (a arrayBase) Dtype() DataType { return a.dtype }

func (a arrayBase) reshape(dims []int) arrayBase {
	n := a.size
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
	return arrayBase{size: n, dims: dims, dtype: a.dtype}
}

func toString(a Array, q Queue) string {
	var data interface{}
	if a.Dtype() == Int32 {
		data = make([]int32, a.Size())
	} else {
		data = make([]float32, a.Size())
	}
	q.Call(Read(a, data)).Finish()
	return format(a.Dims(), data, 0, 1, "", false)
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
func Bytes(arr ...Array) (bytes int) {
	for _, a := range arr {
		if a != nil {
			bytes += 4 * a.Size()
		}
	}
	return bytes
}

// Release one or more arrays
func Release(arr ...Array) {
	for _, a := range arr {
		if a != nil {
			a.Release()
		}
	}
}
