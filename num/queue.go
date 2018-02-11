package num

/*
#include "num.h"
*/
import "C"

import (
	"fmt"
	"github.com/jnb666/deepthought2/num/cuda"
	"github.com/jnb666/deepthought2/num/mkl"
	"runtime"
	"sort"
	"unsafe"
)

// Device interface type
type Device interface {
	// Setup new worker queue
	NewQueue() Queue
	// Allocate new n dimensional array
	NewArray(dtype DataType, dims ...int) *Array
	NewArrayLike(a *Array) *Array
	// Allocate memory with given size in 32 bit words
	NewBuffer(size int) Buffer
}

// Initialise new CPU or GPU device
func NewDevice(useGPU bool) Device {
	if useGPU {
		return gpuDevice{dev: cuda.NewDevice()}
	} else {
		threads := C.int(runtime.NumCPU())
		C.mkl_set_num_threads(threads)
		return cpuDevice{attr: mkl.NewAttr()}
	}
}

// A Queue processes a series of operations on a Device
type Queue interface {
	Device
	Dev() Device
	// Asyncronous function call
	Call(args ...Function) Queue
	// Wait for any pending requests to complete
	Finish()
	// Shutdown the queue and release any resources
	Shutdown()
	// Enable profiling
	Profiling(on bool, title string)
	Profile() string
}

// CPUQueue uses Intel MKL accelarated CPU routines
type cpuDevice struct {
	attr *mkl.Attr
}

type cpuQueue struct {
	cpuDevice
	buffer [C.QUEUE_SIZE]Function
	queued int
	*profile
}

func (d cpuDevice) NewQueue() Queue {
	return &cpuQueue{
		cpuDevice: d,
		profile:   newProfile(false),
	}
}

func (q *cpuQueue) Dev() Device { return q.cpuDevice }

func (q *cpuQueue) exec() {
	bsize := C.int(q.queued)
	ptr := (**C.struct_args)(unsafe.Pointer(&q.buffer[0]))
	var err C.dnnError_t
	var ix C.int
	if q.profile.enabled {
		ix = C.execCPUProfile(bsize, ptr, &err)
		q.profile.add(q.buffer[:q.queued])
	} else {
		ix = C.execCPU(bsize, ptr, &err)
	}
	if ix >= 0 {
		e := mkl.GetError(mkl.Error(err))
		panic(fmt.Sprintf("%s calling %s", e, opDesc(q.buffer[ix])))
	}
	q.queued = 0
}

func (q *cpuQueue) Call(args ...Function) Queue {
	for _, arg := range args {
		if q.queued >= C.QUEUE_SIZE {
			q.exec()
		}
		q.buffer[q.queued] = arg
		q.queued++
	}
	return q
}

func (q *cpuQueue) Finish() {
	if q.queued > 0 {
		q.exec()
	}
}

func (q *cpuQueue) Shutdown() {
	q.Finish()
}

// GPU queue corresponds to a Cuda stream
type gpuDevice struct {
	dev cuda.Device
}

func (d gpuDevice) NewQueue() Queue {
	return &gpuQueue{
		gpuDevice: d,
		profile:   newProfile(true),
		stream:    cuda.NewStream(),
	}
}

type gpuQueue struct {
	gpuDevice
	buffer [C.QUEUE_SIZE]Function
	queued int
	*profile
	stream *cuda.Stream
}

func (q *gpuQueue) Dev() Device { return q.gpuDevice }

func (q *gpuQueue) exec() {
	bsize := C.int(q.queued)
	ptr := (**C.struct_args)(unsafe.Pointer(&q.buffer[0]))
	stream := (*C.struct_stream)(unsafe.Pointer(q.stream))
	var err C.cudaError_t
	var blasErr C.cublasStatus_t
	var dnnErr C.cudnnStatus_t
	var ix C.int
	if q.profile.enabled {
		ix = C.execGPUProfile(bsize, ptr, stream, &err, &blasErr, &dnnErr, &q.profile.events)
		q.profile.add(q.buffer[:q.queued])
	} else {
		ix = C.execGPU(bsize, ptr, stream, &err, &blasErr, &dnnErr)
	}
	if ix < -1 {
		panic(cuda.GetError(cuda.Error(err)))
	}
	if ix >= 0 {
		var e error
		if blasErr != 0 {
			e = cuda.GetBlasError(cuda.BlasStatus(blasErr))
		} else {
			e = cuda.GetDnnError(cuda.DnnStatus(dnnErr))
		}
		panic(fmt.Sprintf("%s calling %s", e, opDesc(q.buffer[ix])))
	}
	q.queued = 0
}

func (q *gpuQueue) Call(args ...Function) Queue {
	for _, arg := range args {
		if q.queued >= C.QUEUE_SIZE {
			q.exec()
		}
		q.buffer[q.queued] = arg
		q.queued++
	}
	return q
}

func (q *gpuQueue) Finish() {
	if q.queued > 0 {
		q.exec()
	}
	q.stream.Sync()
}

func (q *gpuQueue) Shutdown() {
	q.Finish()
	q.stream.Release()
}

// profiling functions
type profile struct {
	prof    map[string]profileRec
	enabled bool
	title   string
	events  C.struct_events
}

type profileRec struct {
	name  string
	calls int64
	msec  float64
}

func newProfile(useGPU bool) *profile {
	p := &profile{prof: make(map[string]profileRec)}
	if useGPU {
		C.initEvents(&p.events)
	}
	return p
}

func (p *profile) Profiling(on bool, title string) {
	p.enabled = on
	p.title = title
}

func (p *profile) add(funcs []Function) {
	for _, f := range funcs {
		name := opDesc(f)
		r := p.prof[name]
		r.name = name
		r.calls++
		r.msec += float64(f.args.msec)
		p.prof[name] = r
	}
}

func (p *profile) Profile() string {
	list := make([]profileRec, len(p.prof))
	i := 0
	for _, v := range p.prof {
		list[i] = v
		i++
	}
	sort.Slice(list, func(i, j int) bool { return list[j].msec < list[i].msec })
	totalCalls := int64(0)
	totalMsec := 0.0
	s := "== " + p.title + " profile ==\n"
	for _, r := range list {
		s += fmt.Sprintf("%-22s %8d calls %10.1f msec\n", r.name, r.calls, r.msec)
		totalCalls += r.calls
		totalMsec += r.msec
	}
	s += fmt.Sprintf("%-22s %8d calls %10.1f msec\n", "TOTAL", totalCalls, totalMsec)
	return s
}
