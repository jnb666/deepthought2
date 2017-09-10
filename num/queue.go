package num

import (
	"runtime"
	"sync"
)

const queueSize = 128

// Device type indicates a compute device, currently only CPU, GPU coming later
type Device int

const (
	CPU Device = iota
)

// A Queue processes a series of operations on a Device
type Queue interface {
	// Device this queue is on
	Device() Device
	// Asyncronous function call
	Call(args ...Function) Queue
	// Wait for any pending requests to complete
	Finish()
	// Shutdown the queue and release any resources
	Shutdown()
}

// NewQueue function starts queue running on the given device
func NewQueue(dev Device, threads int) Queue {
	switch dev {
	case CPU:
		runtime.LockOSThread()
		setCPUThreads(threads)
		return &cpuQueue{}
	default:
		panic("NewQueue: invalid device type")
	}
}

// CPUQueue uses OpenBLAS accelarated CPU routines
type cpuQueue struct {
	buffer [queueSize]Function
	queued int
	sync.Mutex
}

func (a *cpuQueue) Device() Device {
	return CPU
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
