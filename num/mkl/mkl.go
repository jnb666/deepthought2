// Package mkl wraps wraps the Intel MKL DNN functions
package mkl

/*
#cgo CFLAGS: -g -O2 -std=c99 -I/opt/intel/mkl/include
#cgo LDFLAGS: -L/opt/intel/mkl/lib/intel64 -L/opt/intel/tbb/lib/intel64/gcc4.7 -Wl,--no-as-needed -lmkl_intel_lp64 -lmkl_tbb_thread -lmkl_core -ltbb -lstdc++ -lpthread -lm -ldl
#include <mkl.h>
*/
import "C"

import (
	"errors"
	"fmt"
	"github.com/jnb666/deepthought2/num/dnn"
	"runtime"
	"unsafe"
)

var resMap []dnn.ResType

func init() {
	resMap = make([]dnn.ResType, C.dnnResourceNumber)
	for i := dnn.ResType(0); i < dnn.ResNumber; i++ {
		resMap[res(i)] = i
	}
}

func res(t dnn.ResType) C.dnnResourceType_t {
	switch t {
	case dnn.Src:
		return C.dnnResourceSrc
	case dnn.Dst:
		return C.dnnResourceDst
	case dnn.Filter:
		return C.dnnResourceFilter
	case dnn.Bias:
		return C.dnnResourceBias
	case dnn.DiffSrc:
		return C.dnnResourceDiffSrc
	case dnn.DiffDst:
		return C.dnnResourceDiffDst
	case dnn.DiffFilter:
		return C.dnnResourceDiffFilter
	case dnn.DiffBias:
		return C.dnnResourceDiffBias
	case dnn.Workspace:
		return C.dnnResourceWorkspace
	default:
		panic("invalid resource type")
	}
}

type Attr struct {
	h C.dnnPrimitiveAttributes_t
}

func NewAttr() *Attr {
	a := new(Attr)
	chk(C.dnnPrimitiveAttributesCreate_F32(&a.h))
	runtime.SetFinalizer(a, func(obj *Attr) { obj.Release() })
	return a
}

func (a *Attr) Release() {
	C.dnnPrimitiveAttributesDestroy_F32(a.h)
}

// DNN primitive type represents an operation such as a convolution or a conversion
type Primitive struct {
	h C.dnnPrimitive_t
}

func NewPrimitive() *Primitive {
	p := new(Primitive)
	runtime.SetFinalizer(p, func(obj *Primitive) { p.Release() })
	return p
}

func (p *Primitive) Ptr() unsafe.Pointer {
	return unsafe.Pointer(p.h)
}

func (p *Primitive) Release() {
	C.dnnDelete_F32(p.h)
}

// Layer structure represents a DNN layer definition and associated layouts and primitives
type Layer struct {
	Resource
	Fwd, BData     *Primitive
	BFilter, BBias *Primitive
	inShape        []int
	outShape       []int
	filtShape      []int
	biasShape      []int
	name           string
	params         bool
}

func NewLayer(typ string, inShape, outShape []int) *Layer {
	l := &Layer{name: typ, inShape: inShape, outShape: outShape}
	l.Resource = newResources()
	l.Fwd = NewPrimitive()
	l.BData = NewPrimitive()
	return l
}

func (l *Layer) init() {
	l.layoutFromPrimitive(l.Fwd, dnn.Src)
	l.layoutFromPrimitive(l.Fwd, dnn.Dst)
	l.layoutFromPrimitive(l.BData, dnn.DiffSrc)
	l.layoutFromPrimitive(l.BData, dnn.DiffDst)
	l.Alloc(dnn.Dst)
	l.Alloc(dnn.DiffSrc)
}

func (l *Layer) initParams() {
	l.layoutFromPrimitive(l.Fwd, dnn.Filter)
	l.layoutFromPrimitive(l.Fwd, dnn.Bias)
	l.layoutFromPrimitive(l.BFilter, dnn.DiffFilter)
	l.layoutFromPrimitive(l.BBias, dnn.DiffBias)
}

func (l *Layer) layoutFromPrimitive(prim *Primitive, typ dnn.ResType) {
	t := res(typ)
	layout := new(Layout)
	chk(C.dnnLayoutCreateFromPrimitive_F32(&layout.h, prim.h, t))
	l.layout[t] = layout
	runtime.SetFinalizer(layout, func(obj *Layout) { obj.Release() })
}

func (l *Layer) Alloc(typ dnn.ResType) {
	t := res(typ)
	l.data[t] = dnn.Alloc(l.layout[t].Size())
}

// Set a conversion from input layout to given layout, returns false if input and output have same layout
func (l *Layer) InitInConv(typ dnn.ResType, dims []int, order dnn.DataLayout) bool {
	t := res(typ)
	l.inConv[t] = NewLayout(dims, order).Conversion(l.layout[t])
	return l.inConv != nil
}

// Setup a conversion from given resource to output layout, returns false if input and output have same layout
func (l *Layer) InitOutConv(typ dnn.ResType, dims []int, order dnn.DataLayout) bool {
	t := res(typ)
	l.outConv[t] = l.layout[t].Conversion(NewLayout(dims, order))
	return l.outConv != nil
}

func (l *Layer) Shape(typ dnn.ResType) []int {
	switch typ {
	case dnn.Src, dnn.DiffSrc:
		return l.inShape
	case dnn.Dst, dnn.DiffDst:
		return l.outShape
	case dnn.Filter, dnn.DiffFilter:
		return l.filtShape
	case dnn.Bias, dnn.DiffBias:
		return l.biasShape
	default:
		panic("invalid type")
	}
}

func (l *Layer) String() string {
	return fmt.Sprintf("[%s]  inShape=%v  outShape=%v\n\t%s\n", l.name, l.inShape, l.outShape, l.Resource)
}

func (l *Layer) Type() string {
	return l.name
}

func (l *Layer) HasParams() bool {
	return l.BFilter != nil
}

// Setup new linear layer
func InnerProduct(attr *Attr, nBatch, nIn, nOut int, order dnn.DataLayout) *Layer {
	l := NewLayer("linear", []int{nIn, nBatch}, []int{nOut, nBatch})
	l.filtShape = []int{nIn, nOut}
	l.biasShape = []int{nOut}
	l.BFilter = NewPrimitive()
	l.BBias = NewPrimitive()
	inSize := sizeDims(l.inShape, order)
	outSize := sizeDims(l.outShape, order)
	chans := C.size_t(nOut)
	chk(C.dnnInnerProductCreateForwardBias_F32(&l.Fwd.h, attr.h, 2, &inSize[0], chans))
	chk(C.dnnInnerProductCreateBackwardData_F32(&l.BData.h, attr.h, 2, &inSize[0], chans))
	chk(C.dnnInnerProductCreateBackwardFilter_F32(&l.BFilter.h, attr.h, 2, &inSize[0], chans))
	chk(C.dnnInnerProductCreateBackwardBias_F32(&l.BBias.h, attr.h, 2, &outSize[0]))
	l.init()
	l.initParams()
	return l
}

// Setup new convolution layer
func Convolution(attr *Attr, n, d, h, w, nFeats, filtSize, stride, pad int, order dnn.DataLayout) *Layer {
	wOut := outSize(w, filtSize, stride, pad)
	hOut := outSize(h, filtSize, stride, pad)
	l := NewLayer("conv", []int{w, h, d, n}, []int{wOut, hOut, nFeats, n})
	l.filtShape = []int{filtSize, filtSize, d, nFeats}
	l.biasShape = []int{nFeats}
	l.BFilter = NewPrimitive()
	l.BBias = NewPrimitive()
	inSize := sizeDims(l.inShape, order)
	outSize := sizeDims(l.outShape, order)
	filter := sizeDims(l.filtShape, order)
	cstride := sizeDims2(stride)
	offset := [2]C.int{C.int(-pad), C.int(-pad)}
	chk(C.dnnConvolutionCreateForwardBias_F32(&l.Fwd.h, attr.h, C.dnnAlgorithmConvolutionDirect,
		4, &inSize[0], &outSize[0], &filter[0], &cstride[0], &offset[0], C.dnnBorderZeros))
	chk(C.dnnConvolutionCreateBackwardData_F32(&l.BData.h, attr.h, C.dnnAlgorithmConvolutionDirect,
		4, &inSize[0], &outSize[0], &filter[0], &cstride[0], &offset[0], C.dnnBorderZeros))
	chk(C.dnnConvolutionCreateBackwardFilter_F32(&l.BFilter.h, attr.h, C.dnnAlgorithmConvolutionDirect,
		4, &inSize[0], &outSize[0], &filter[0], &cstride[0], &offset[0], C.dnnBorderZeros))
	chk(C.dnnConvolutionCreateBackwardBias_F32(&l.BBias.h, attr.h, C.dnnAlgorithmConvolutionDirect,
		4, &outSize[0]))
	l.init()
	l.initParams()
	return l
}

// Setup new max pooling layer
func MaxPooling(attr *Attr, prev *Layer, size, stride int) *Layer {
	inShape := prev.outShape
	wOut := outSize(inShape[0], size, stride, 0)
	hOut := outSize(inShape[1], size, stride, 0)
	l := NewLayer("maxPool", inShape, []int{wOut, hOut, inShape[2], inShape[3]})
	in := prev.GetLayout(dnn.Dst)
	csize := sizeDims2(size)
	cstride := sizeDims2(stride)
	offset := [2]C.int{}
	chk(C.dnnPoolingCreateForward_F32(&l.Fwd.h, attr.h, C.dnnAlgorithmPoolingMax,
		in.h, &csize[0], &cstride[0], &offset[0], C.dnnBorderZeros))
	chk(C.dnnPoolingCreateBackward_F32(&l.BData.h, attr.h, C.dnnAlgorithmPoolingMax,
		in.h, &csize[0], &cstride[0], &offset[0], C.dnnBorderZeros))
	l.init()
	l.layoutFromPrimitive(l.Fwd, dnn.Workspace)
	l.Alloc(dnn.Workspace)
	return l
}

// Setup new relu activation layer
func Relu(attr *Attr, prev *Layer) *Layer {
	shape := prev.outShape
	l := NewLayer("relu", shape, shape)
	in := prev.GetLayout(dnn.Dst)
	out := NewLayout(prev.Shape(dnn.Dst), dnn.ColMajor)
	chk(C.dnnReLUCreateForward_F32(&l.Fwd.h, attr.h, in.h, 0))
	chk(C.dnnReLUCreateBackward_F32(&l.BData.h, attr.h, out.h, in.h, 0))
	l.init()
	return l
}

// Resources associated with layer
type Resource struct {
	data    []unsafe.Pointer
	inConv  []*Primitive
	outConv []*Primitive
	layout  []*Layout
}

func newResources() Resource {
	return Resource{
		data:    make([]unsafe.Pointer, C.dnnResourceNumber),
		layout:  make([]*Layout, C.dnnResourceNumber),
		inConv:  make([]*Primitive, C.dnnResourceNumber),
		outConv: make([]*Primitive, C.dnnResourceNumber),
	}
}

func (r Resource) ResPtr() unsafe.Pointer {
	return unsafe.Pointer(&r.data[0])
}

func (r Resource) Data(typ dnn.ResType) unsafe.Pointer {
	return r.data[res(typ)]
}

func (r Resource) SetData(typ dnn.ResType, p unsafe.Pointer) {
	r.data[res(typ)] = p
}

func (r Resource) GetLayout(typ dnn.ResType) *Layout {
	return r.layout[res(typ)]
}

func (r Resource) InConv(typ dnn.ResType) *Primitive {
	return r.inConv[res(typ)]
}

func (r Resource) OutConv(typ dnn.ResType) *Primitive {
	return r.outConv[res(typ)]
}

func (r Resource) String() string {
	s := ""
	for i, ptr := range r.data {
		if ptr != nil {
			s += resMap[i].String()
			if r.inConv[i] != nil {
				s += "(in)"
			}
			if r.outConv[i] != nil {
				s += "(out)"
			}
			s += " "
		}
	}
	return s
}

// DNN layout container reprents a n dimensional tensor
type Layout struct {
	dims []int
	h    C.dnnLayout_t
}

func NewLayout(dims []int, order dnn.DataLayout) *Layout {
	l := new(Layout)
	ndim := C.size_t(len(dims))
	size := sizeDims(dims, order)
	strides := getStrides(dims, order)
	chk(C.dnnLayoutCreate_F32(&l.h, ndim, &size[0], &strides[0]))
	runtime.SetFinalizer(l, func(obj *Layout) { obj.Release() })
	return l
}

func (l *Layout) Dims() []int {
	return l.dims
}

func (l *Layout) Release() {
	C.dnnLayoutDelete_F32(l.h)
}

// Setup a conversion to copy data to the given layout returns nil if output layerout is the same
func (l *Layout) Conversion(to *Layout) *Primitive {
	if C.dnnLayoutCompare_F32(l.h, l.h) == 0 {
		return nil
	}
	conv := NewPrimitive()
	chk(C.dnnConversionCreate_F32(&conv.h, l.h, to.h))
	return conv
}

// Get size of layout in 32 bit words
func (l *Layout) Size() int {
	size := C.dnnLayoutGetMemorySize_F32(l.h)
	return int(size) / 4
}

// utilities
func outSize(x, size, stride, pad int) int {
	ns := x - size + 2*pad
	if ns%stride != 0 {
		panic("output size invalid, must be even no. of strides")
	}
	return ns/stride + 1
}

func sizeDims2(a int) [2]C.size_t {
	return [2]C.size_t{C.size_t(a), C.size_t(a)}
}

func sizeDims(dims []int, order dnn.DataLayout) []C.size_t {
	res := make([]C.size_t, len(dims))
	for i, dim := range dims {
		if order == dnn.RowMajor {
			res[len(dims)-i-1] = C.size_t(dim)
		} else {
			res[i] = C.size_t(dim)
		}
	}
	return res
}

func getStrides(dims []int, order dnn.DataLayout) []C.size_t {
	p := 1
	res := make([]C.size_t, len(dims))
	for i, dim := range dims {
		if order == dnn.RowMajor {
			res[len(dims)-i-1] = C.size_t(p)
		} else {
			res[i] = C.size_t(p)
		}
		p *= dim
	}
	return res
}

type Error C.dnnError_t

// Check for error running DNN function, panics if not success
func GetError(err Error) error {
	return getError(C.dnnError_t(err))
}

func chk(err C.dnnError_t) {
	errDesc := getError(err)
	if errDesc != nil {
		panic(errDesc)
	}
}

func getError(err C.dnnError_t) error {
	switch err {
	case C.E_SUCCESS:
		return nil
	case C.E_INCORRECT_INPUT_PARAMETER:
		return errors.New("MKL_DNN: incorrect input parameter")
	case C.E_MEMORY_ERROR:
		return errors.New("MKL_DNN: memory allocation failed")
	case C.E_UNSUPPORTED_DIMENSION:
		return errors.New("MKL_DNN: unsupported dimension")
	case C.E_UNIMPLEMENTED:
		return errors.New("MKL_DNN: not implemented")
	default:
		return errors.New("MKL_DNN: unknown error!")
	}
}
