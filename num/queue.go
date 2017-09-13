package num

import (
	"github.com/jnb666/deepthought2/num/dnn"
	"github.com/jnb666/deepthought2/num/mkl"
	"runtime"
	"sync"
)

const queueSize = 128

// Device type indicates a compute device, currently only CPU, GPU coming later
type DeviceType int

const (
	CPU DeviceType = iota
)

// Device interface type
type Device interface {
	// Setup new worker queue
	NewQueue(threads int) Queue
	// Allocate new n dimensional array
	NewArray(dtype DataType, dims ...int) Array
	NewArrayLike(a Array) Array
	// Allocate DNN layer
	LinearLayer(nBatch, nIn, nOut int) dnn.Layer
	ReluLayer(prev dnn.Layer) dnn.Layer
}

type cpuDevice struct {
	attr *mkl.Attr
}

// Initialise new CPU device
func NewCPUDevice() Device {
	return cpuDevice{attr: mkl.NewAttr()}
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
}

// CPUQueue uses OpenBLAS accelarated CPU routines
type cpuQueue struct {
	cpuDevice
	buffer [queueSize]Function
	queued int
	sync.Mutex
}

func (d cpuDevice) NewQueue(threads int) Queue {
	runtime.LockOSThread()
	setCPUThreads(threads)
	return &cpuQueue{cpuDevice: d}
}

func (q *cpuQueue) Dev() Device {
	return q.cpuDevice
}

func (q *cpuQueue) Call(args ...Function) Queue {
	q.Lock()
	defer q.Unlock()
	for _, arg := range args {
		if q.queued >= queueSize {
			execCPU(q.buffer[:q.queued])
			q.queued = 0
		}
		q.buffer[q.queued] = arg
		q.queued++
	}
	return q
}

func (q *cpuQueue) Finish() {
	if q.queued > 0 {
		execCPU(q.buffer[:q.queued])
		q.queued = 0
	}
}

func (q *cpuQueue) Shutdown() {
	q.Finish()
}
