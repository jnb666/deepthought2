package num

import (
	"fmt"
	"github.com/jnb666/deepthought2/num/mkl"
	"runtime"
	"sort"
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
	// Enable profiling
	Profiling(on bool)
	PrintProfile()
}

// CPUQueue uses OpenBLAS accelarated CPU routines
type cpuQueue struct {
	cpuDevice
	buffer  [queueSize]Function
	queued  int
	prof    map[string]profileRec
	profile bool
	sync.Mutex
}

type profileRec struct {
	name  string
	calls int64
	usec  int64
}

func (d cpuDevice) NewQueue(threads int) Queue {
	runtime.LockOSThread()
	setCPUThreads(threads)
	return &cpuQueue{cpuDevice: d, prof: make(map[string]profileRec)}
}

func (q *cpuQueue) Dev() Device {
	return q.cpuDevice
}

func (q *cpuQueue) Call(args ...Function) Queue {
	q.Lock()
	defer q.Unlock()
	for _, arg := range args {
		if q.queued >= queueSize {
			execCPU(q.buffer[:q.queued], q.profile, q.prof)
			q.queued = 0
		}
		q.buffer[q.queued] = arg
		q.queued++
	}
	return q
}

func (q *cpuQueue) Finish() {
	if q.queued > 0 {
		execCPU(q.buffer[:q.queued], q.profile, q.prof)
		q.queued = 0
	}
}

func (q *cpuQueue) Shutdown() {
	q.Finish()
	if q.profile {
		q.PrintProfile()
	}
}

func (q *cpuQueue) Profiling(on bool) {
	q.profile = on
}

func (q *cpuQueue) PrintProfile() {
	fmt.Println("== Profile ==")
	list := make([]profileRec, len(q.prof))
	i := 0
	for _, v := range q.prof {
		list[i] = v
		i++
	}
	sort.Slice(list, func(i, j int) bool { return list[j].usec < list[i].usec })
	for _, r := range list {
		fmt.Printf("%-20s %8d calls %10d usec\n", r.name, r.calls, r.usec)
	}
}
