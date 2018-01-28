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
}

func NewLayer(typ string, inShape, outShape []int) *Layer {
	l := &Layer{name: typ, inShape: inShape, outShape: outShape}
	l.Resource = newResources()
	l.Fwd = NewPrimitive()
	l.BData = NewPrimitive()
	return l
}

func (l *Layer) Release() {
	for _, p := range []*Primitive{l.Fwd, l.BData, l.BFilter, l.BBias} {
		if p != nil {
			p.Release()
		}
	}
	for _, l := range l.layout {
		if l != nil {
			l.Release()
		}
	}
}

func (l *Layer) init() {
	l.layoutFromPrimitive(l.Fwd, C.dnnResourceSrc)
	l.layoutFromPrimitive(l.Fwd, C.dnnResourceDst)
	l.layoutFromPrimitive(l.BData, C.dnnResourceDiffSrc)
	l.layoutFromPrimitive(l.BData, C.dnnResourceDiffDst)
	l.Alloc(C.dnnResourceDst)
}

func (l *Layer) initParams(noBias bool) {
	l.layoutFromPrimitive(l.Fwd, C.dnnResourceFilter)
	l.layoutFromPrimitive(l.BFilter, C.dnnResourceDiffFilter)
	if !noBias {
		l.layoutFromPrimitive(l.Fwd, C.dnnResourceBias)
		l.layoutFromPrimitive(l.BBias, C.dnnResourceDiffBias)
	}
}

func (l *Layer) layoutFromPrimitive(prim *Primitive, typ C.dnnResourceType_t) {
	layout := new(Layout)
	chk(C.dnnLayoutCreateFromPrimitive_F32(&layout.h, prim.h, typ))
	l.layout[typ] = layout
}

func (l *Layer) Alloc(typ C.dnnResourceType_t) {
	buf := NewBuffer(l.layout[typ].Size())
	l.data[typ] = buf.Data()
}

func (l *Layer) String() string {
	return fmt.Sprintf("[%s]  inShape=%v  outShape=%v\n\t%s\n", l.name, l.inShape, l.outShape, l.Resource)
}

func (l *Layer) InShape() []int { return l.inShape }

func (l *Layer) OutShape() []int { return l.outShape }

func (l *Layer) FilterShape() []int { return l.filtShape }

func (l *Layer) BiasShape() []int { return l.biasShape }

func (l *Layer) Type() string { return l.name }

func (l *Layer) Worksize() int {
	if l.layout[C.dnnResourceWorkspace] == nil {
		return 0
	}
	return l.layout[C.dnnResourceWorkspace].Size()
}

// Setup new convolution layer
func Convolution(attr *Attr, n, c, h, w, nFeats, filtSize, stride int, padding, noBias bool) *Layer {
	wOut, wPad, err := getOutSize(w, filtSize, stride, padding)
	if err != nil {
		panic(err)
	}
	hOut, hPad, err := getOutSize(h, filtSize, stride, padding)
	if err != nil {
		panic(err)
	}
	l := NewLayer("conv", []int{w, h, c, n}, []int{wOut, hOut, nFeats, n})
	l.filtShape = []int{filtSize, filtSize, c, nFeats}
	inSize := sizeDims(l.inShape)
	outSize := sizeDims(l.outShape)
	filter := sizeDims(l.filtShape)
	cstride := sizeDims2(stride)
	btype, offset := getOffset(hPad, wPad)
	l.BFilter = NewPrimitive()
	if noBias {
		chk(C.dnnConvolutionCreateForward_F32(&l.Fwd.h, attr.h, C.dnnAlgorithmConvolutionDirect,
			4, &inSize[0], &outSize[0], &filter[0], &cstride[0], &offset[0], btype))
	} else {
		l.biasShape = []int{1, 1, nFeats, 1}
		l.BBias = NewPrimitive()
		chk(C.dnnConvolutionCreateForwardBias_F32(&l.Fwd.h, attr.h, C.dnnAlgorithmConvolutionDirect,
			4, &inSize[0], &outSize[0], &filter[0], &cstride[0], &offset[0], btype))
		chk(C.dnnConvolutionCreateBackwardBias_F32(&l.BBias.h, attr.h, C.dnnAlgorithmConvolutionDirect,
			4, &outSize[0]))
	}
	chk(C.dnnConvolutionCreateBackwardData_F32(&l.BData.h, attr.h, C.dnnAlgorithmConvolutionDirect,
		4, &inSize[0], &outSize[0], &filter[0], &cstride[0], &offset[0], btype))
	chk(C.dnnConvolutionCreateBackwardFilter_F32(&l.BFilter.h, attr.h, C.dnnAlgorithmConvolutionDirect,
		4, &inSize[0], &outSize[0], &filter[0], &cstride[0], &offset[0], btype))
	l.init()
	l.initParams(noBias)
	return l
}

// Setup new max pooling or average pooling layer
func Pooling(attr *Attr, n, c, h, w, size, stride int, padding, average bool) *Layer {
	wOut, wPad, _ := getOutSize(w, size, stride, padding)
	hOut, hPad, _ := getOutSize(h, size, stride, padding)
	l := NewLayer("pool", []int{w, h, c, n}, []int{wOut, hOut, c, n})
	csize := sizeDims2(size)
	cstride := sizeDims2(stride)
	btype, offset := getOffset(hPad, wPad)
	in := NewLayout(l.inShape)
	var algo C.dnnAlgorithm_t
	if average {
		algo = C.dnnAlgorithmPoolingAvgExcludePadding
	} else {
		algo = C.dnnAlgorithmPoolingMax
	}
	chk(C.dnnPoolingCreateForward_F32(&l.Fwd.h, attr.h, algo, in.h, &csize[0], &cstride[0], &offset[0], btype))
	chk(C.dnnPoolingCreateBackward_F32(&l.BData.h, attr.h, algo, in.h, &csize[0], &cstride[0], &offset[0], btype))
	l.init()
	l.layoutFromPrimitive(l.Fwd, C.dnnResourceWorkspace)
	l.Alloc(C.dnnResourceWorkspace)
	return l
}

// Setup new batch normalisation layer
func BatchNorm(attr *Attr, n, c, h, w int, epsilon float64) *Layer {
	l := NewLayer("batchNorm", []int{w, h, c, n}, []int{w, h, c, n})
	l.filtShape = []int{c, 2}
	in := NewLayout(l.inShape)
	eps := C.float(epsilon)
	chk(C.dnnBatchNormalizationCreateForward_v2_F32(&l.Fwd.h, attr.h, in.h, eps, C.dnnUseScaleShift))
	chk(C.dnnBatchNormalizationCreateBackward_v2_F32(&l.BData.h, attr.h, in.h, eps, C.dnnUseScaleShift))
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

func (r Resource) Dst() Buffer {
	return Buffer{ptr: r.data[C.dnnResourceDst], size: r.layout[C.dnnResourceDst].Size()}
}

func (r Resource) SetSrc(p unsafe.Pointer) {
	r.data[C.dnnResourceSrc] = p
}

func (r Resource) SetDiffSrc(p unsafe.Pointer) {
	r.data[C.dnnResourceDiffSrc] = p
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

func (r Resource) SetStatsData(w, dw, runMean, runVar unsafe.Pointer) {
	r.data[C.dnnResourceScaleShift] = w
	r.data[C.dnnResourceDiffScaleShift] = dw
	r.data[C.dnnResourceMean] = runMean
	r.data[C.dnnResourceVariance] = runVar
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
	Dims []int
	h    C.dnnLayout_t
}

func NewLayout(dims []int) *Layout {
	l := &Layout{Dims: dims}
	ndim := C.size_t(len(dims))
	size := sizeDims(dims)
	strides := getStrides(dims)
	chk(C.dnnLayoutCreate_F32(&l.h, ndim, &size[0], &strides[0]))
	return l
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
func getOutSize(in, filter, stride int, padding bool) (out, pad int, err error) {
	if filter > in {
		err = fmt.Errorf("filter size %d > input size %d", filter, in)
		return
	}
	var end int
	if padding {
		out = in / stride
		end = filter + (out-1)*stride
		if end < in {
			out++
			end += stride
		}
		pad = (end - in) / 2
		if (end-in)%2 != 0 {
			pad++
		}
	} else {
		out = 1 + (in-filter)/stride
		end = filter + (out-1)*stride
		if end != in {
			err = fmt.Errorf("filter %d and stride %d does not divide input %d", filter, stride, in)
		}
	}
	return
}

func getOffset(hPad, wPad int) (btype C.dnnBorder_t, offset [2]C.int) {
	btype = C.dnnBorderZeros
	offset[0] = C.int(-wPad)
	offset[1] = C.int(-hPad)
	return
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

// Convert MKL DNN error code to go error
func GetError(err Error) error {
	return getError(C.dnnError_t(err))
}

// Check for error running DNN function, panics if not success
func chk(err C.dnnError_t) {
	if e := getError(err); e != nil {
		panic(e)
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

type Buffer struct {
	ptr  unsafe.Pointer
	size int
}

// Allocate a block of memory of given no. of 32 bit words - align on 64 byte boundary
func NewBuffer(size int) Buffer {
	if size <= 0 {
		panic("NewBuffer: size must be greater than 0")
	}
	b := Buffer{size: size}
	arr := make([]float32, size+16)
	b.ptr = unsafe.Pointer(&arr[0])
	off := (uintptr(b.ptr) % 64) / 4
	if off != 0 {
		b.ptr = unsafe.Pointer(&arr[16-off])
	}
	return b
}

func (b Buffer) Data() unsafe.Pointer {
	return b.ptr
}

func (b Buffer) Size() int {
	return b.size
}

func (b Buffer) Release() {
	if b.size > 0 {
		b.size = 0
		b.ptr = nil
	}
}
