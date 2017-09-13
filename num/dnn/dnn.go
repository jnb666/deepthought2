// Package dnn has generic types for wrapping deep neural network functions, e.g. MKL, Cuda etc.
package dnn

import "unsafe"

type DataLayout bool

const (
	RowMajor DataLayout = false
	ColMajor DataLayout = true
)

type ResType int

const (
	Src ResType = iota
	Dst
	Filter
	Bias
	DiffSrc
	DiffDst
	DiffFilter
	DiffBias
	ResNumber
)

var resNames = map[ResType]string{
	Src:        "Src",
	Dst:        "Dst",
	Filter:     "Filter",
	Bias:       "Bias",
	DiffSrc:    "DiffSrc",
	DiffDst:    "DiffDst",
	DiffFilter: "DiffFilter",
	DiffBias:   "DiffBias",
}

func (r ResType) String() string {
	return resNames[r]
}

// Layer interface type represents a DNN layer
type Layer interface {
	InShape() []int
	OutShape() []int
	Data(typ ResType) unsafe.Pointer
	SetData(typ ResType, p unsafe.Pointer)
}

// Allocate a block of memory of given no. of 32 bit words - align on 64 byte boundary
func Alloc(size int) unsafe.Pointer {
	buf := make([]float32, size+16)
	ptr := unsafe.Pointer(&buf[0])
	off := (uintptr(ptr) % 64) / 4
	if off != 0 {
		return unsafe.Pointer(&buf[16-off])
	} else {
		return ptr
	}
}
