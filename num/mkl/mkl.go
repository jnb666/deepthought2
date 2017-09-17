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
	"runtime"
	"unsafe"
)

var resNames = map[C.dnnResourceType_t]string{
	C.dnnResourceSrc:        "Src",
	C.dnnResourceDst:        "Dst",
	C.dnnResourceFilter:     "Filter",
	C.dnnResourceBias:       "Bias",
	C.dnnResourceDiffSrc:    "DiffSrc",
	C.dnnResourceDiffDst:    "DiffDst",
	C.dnnResourceDiffFilter: "DiffFilter",
	C.dnnResourceDiffBias:   "DiffBias",
	C.dnnResourceWorkspace:  "WorkSpace",
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
	l.layoutFromPrimitive(l.Fwd, C.dnnResourceSrc)
	l.layoutFromPrimitive(l.Fwd, C.dnnResourceDst)
	l.layoutFromPrimitive(l.BData, C.dnnResourceDiffSrc)
	l.layoutFromPrimitive(l.BData, C.dnnResourceDiffDst)
	l.Alloc(C.dnnResourceDst)
	l.Alloc(C.dnnResourceDiffSrc)
}

func (l *Layer) initParams() {
	l.layoutFromPrimitive(l.Fwd, C.dnnResourceFilter)
	l.layoutFromPrimitive(l.Fwd, C.dnnResourceBias)
	l.layoutFromPrimitive(l.BFilter, C.dnnResourceDiffFilter)
	l.layoutFromPrimitive(l.BBias, C.dnnResourceDiffBias)
}

func (l *Layer) layoutFromPrimitive(prim *Primitive, typ C.dnnResourceType_t) {
	layout := new(Layout)
	chk(C.dnnLayoutCreateFromPrimitive_F32(&layout.h, prim.h, typ))
	l.layout[typ] = layout
	runtime.SetFinalizer(layout, func(obj *Layout) { obj.Release() })
}

func (l *Layer) Alloc(typ C.dnnResourceType_t) {
	l.data[typ] = NewBuffer(l.layout[typ].Size())
}

func (l *Layer) String() string {
	return fmt.Sprintf("[%s]  inShape=%v  outShape=%v\n\t%s\n", l.name, l.inShape, l.outShape, l.Resource)
}

func (l *Layer) InShape() []int { return l.inShape }

func (l *Layer) OutShape() []int { return l.outShape }

func (l *Layer) FilterShape() []int { return l.filtShape }

func (l *Layer) BiasShape() []int { return l.biasShape }

func (l *Layer) Type() string { return l.name }

func (l *Layer) HasParams() bool { return l.BFilter != nil }

// Setup new linear layer
func InnerProduct(attr *Attr, nBatch, nIn, nOut int) *Layer {
	l := NewLayer("linear", []int{nIn, nBatch}, []int{nOut, nBatch})
	l.filtShape = []int{nIn, nOut}
	l.biasShape = []int{nOut}
	l.BFilter = NewPrimitive()
	l.BBias = NewPrimitive()
	inSize := sizeDims(l.inShape)
	outSize := sizeDims(l.outShape)
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
func Convolution(attr *Attr, n, d, h, w, nFeats, filtSize, stride, pad int) *Layer {
	wOut := outSize(w, filtSize, stride, pad)
	hOut := outSize(h, filtSize, stride, pad)
	l := NewLayer("conv", []int{w, h, d, n}, []int{wOut, hOut, nFeats, n})
	l.filtShape = []int{filtSize, filtSize, d, nFeats}
	l.biasShape = []int{nFeats}
	l.BFilter = NewPrimitive()
	l.BBias = NewPrimitive()
	inSize := sizeDims(l.inShape)
	outSize := sizeDims(l.outShape)
	filter := sizeDims(l.filtShape)
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
	in := prev.layout[C.dnnResourceDst]
	csize := sizeDims2(size)
	cstride := sizeDims2(stride)
	offset := [2]C.int{}
	chk(C.dnnPoolingCreateForward_F32(&l.Fwd.h, attr.h, C.dnnAlgorithmPoolingMax,
		in.h, &csize[0], &cstride[0], &offset[0], C.dnnBorderZeros))
	chk(C.dnnPoolingCreateBackward_F32(&l.BData.h, attr.h, C.dnnAlgorithmPoolingMax,
		in.h, &csize[0], &cstride[0], &offset[0], C.dnnBorderZeros))
	l.init()
	l.layoutFromPrimitive(l.Fwd, C.dnnResourceWorkspace)
	l.Alloc(C.dnnResourceWorkspace)
	return l
}

// Setup new relu activation layer
func Relu(attr *Attr, prev *Layer) *Layer {
	shape := prev.outShape
	l := NewLayer("relu", shape, shape)
	in := prev.layout[C.dnnResourceDst]
	out := NewLayout(prev.outShape)
	chk(C.dnnReLUCreateForward_F32(&l.Fwd.h, attr.h, in.h, 0))
	chk(C.dnnReLUCreateBackward_F32(&l.BData.h, attr.h, out.h, in.h, 0))
	l.init()
	return l
}

// Resources associated with layer
type Resource struct {
	data   []unsafe.Pointer
	layout []*Layout
}

func newResources() Resource {
	return Resource{
		data:   make([]unsafe.Pointer, C.dnnResourceNumber),
		layout: make([]*Layout, C.dnnResourceNumber),
	}
}

func (r Resource) ResPtr() unsafe.Pointer {
	return unsafe.Pointer(&r.data[0])
}

func (r Resource) Dst() unsafe.Pointer {
	return r.data[C.dnnResourceDst]
}

func (r Resource) DiffSrc() unsafe.Pointer {
	return r.data[C.dnnResourceDiffSrc]
}

func (r Resource) SetSrc(p unsafe.Pointer) {
	r.data[C.dnnResourceSrc] = p
}

func (r Resource) SetDiffDst(p unsafe.Pointer) {
	r.data[C.dnnResourceDiffDst] = p
}

func (r Resource) SetParams(W, B, dW, dB unsafe.Pointer) {
	r.data[C.dnnResourceFilter] = W
	r.data[C.dnnResourceBias] = B
	r.data[C.dnnResourceDiffFilter] = dW
	r.data[C.dnnResourceDiffBias] = dB
}

func (r Resource) String() string {
	s := ""
	for i, ptr := range r.data {
		if ptr != nil {
			s += resNames[C.dnnResourceType_t(i)] + " "
		}
	}
	return s
}

// DNN layout container reprents a n dimensional tensor
type Layout struct {
	dims []int
	h    C.dnnLayout_t
}

func NewLayout(dims []int) *Layout {
	l := new(Layout)
	ndim := C.size_t(len(dims))
	size := sizeDims(dims)
	strides := getStrides(dims)
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

func sizeDims(dims []int) []C.size_t {
	res := make([]C.size_t, len(dims))
	for i, dim := range dims {
		res[i] = C.size_t(dim)
	}
	return res
}

func getStrides(dims []int) []C.size_t {
	p := 1
	res := make([]C.size_t, len(dims))
	for i, dim := range dims {
		res[i] = C.size_t(p)
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

// Allocate a block of memory of given no. of 32 bit words - align on 64 byte boundary
func NewBuffer(size int) unsafe.Pointer {
	buf := make([]float32, size+16)
	ptr := unsafe.Pointer(&buf[0])
	off := (uintptr(ptr) % 64) / 4
	if off != 0 {
		return unsafe.Pointer(&buf[16-off])
	} else {
		return ptr
	}
}
