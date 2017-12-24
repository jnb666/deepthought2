package nnet

import (
	"github.com/jnb666/deepthought2/num"
	"math/rand"
	"testing"
)

const (
	batch = 5
	nIn   = 6
	nOut  = 4
	eps   = 1e-6
)

var devices []num.Device

func init() {
	devices = []num.Device{
		num.NewDevice(false),
		num.NewDevice(true),
	}
}

func abs(x float32) float32 {
	if x >= 0 {
		return x
	}
	return -x
}

func randArray(size int, min, max float32) []float32 {
	v := make([]float32, size)
	for i := range v {
		v[i] = min + rand.Float32()*(max-min)
	}
	return v
}

func getInputs(t *testing.T, q num.Queue) (input, W, B num.Array) {
	rand.Seed(42)
	input = q.NewArray(num.Float32, nIn, batch)
	W = q.NewArray(num.Float32, nIn, nOut)
	B = q.NewArray(num.Float32, nOut)
	weights := randArray(nIn*nOut, -0.5, 0.5)
	bias := randArray(nOut, 0.1, 0.2)
	inData := randArray(batch*nIn, 0, 1)
	q.Call(
		num.Write(W, weights),
		num.Write(B, bias),
		num.Write(input, inData),
	)
	t.Logf("== weights ==\n%s %s", W.String(q), B.String(q))
	t.Logf("== input ==\n%s", input.String(q))
	return
}

func setupNetwork(q num.Queue, W, B num.Array) (l1, l2 Layer, dW, dB num.Array) {
	lin := &linear{Linear: Linear{Nout: nOut}}
	lin.Init(q, []int{nIn, batch}, 0, nil)
	lin.SetParams(q, W, B)
	relu := &activation{Activation: Activation{Atype: "relu"}}
	relu.Init(q, []int{nOut, batch}, 1, nil)
	return lin, relu, lin.dw, lin.db
}

func compareArray(t *testing.T, q num.Queue, title string, A num.Array, expect []float32) {
	t.Logf("== %s DNN ==\n%s", title, A.String(q))
	arr := make([]float32, num.Prod(A.Dims()))
	q.Call(num.Read(A, arr)).Finish()
	if len(arr) != len(expect) {
		t.Fatal(title, "length mismatch!")
	}
	for i := range arr {
		if abs(arr[i]-expect[i]) > eps {
			t.Error(title, "mismatch!")
			return
		}
	}
}

func TestFprop(t *testing.T) {
	for _, dev := range devices {
		q := dev.NewQueue()
		input, W, B := getInputs(t, q)
		lin, relu, _, _ := setupNetwork(q, W, B)

		temp := lin.Fprop(q, input, nil, true)
		output := relu.Fprop(q, temp, nil, true)

		expect := []float32{
			0, 0.21253887, 0.49112207, 0,
			0, 0, 0.10656603, 0,
			0, 0.23254871, 0.11656132, 0.38451424,
			0, 0.3471108, 0.013840735, 0.3113696,
			0, 0.3044004, 0, 0.48620278}

		compareArray(t, q, "output", output, expect)
		q.Shutdown()
	}
}

func getOutput(q num.Queue) num.Array {
	data := make([]float32, nOut*batch)
	for row := 0; row < batch; row++ {
		data[row+rand.Intn(nOut)*batch] = 1
	}
	yOneHot := q.NewArray(num.Float32, nOut, batch)
	q.Call(num.Write(yOneHot, data))
	return yOneHot
}

func inputGrad(t *testing.T, q num.Queue, output, yOneHot num.Array) num.Array {
	inGrad := q.NewArrayLike(output)
	q.Call(
		num.Copy(output, inGrad),
		num.Axpy(-1, yOneHot, inGrad),
	)
	t.Logf("== input grad ==\n%s", inGrad.String(q))
	return inGrad
}

func TestDNNBprop(t *testing.T) {
	for _, dev := range devices {
		q := dev.NewQueue()
		input, W, B := getInputs(t, q)
		lin, relu, dW, dB := setupNetwork(q, W, B)
		yOneHot := getOutput(q)

		temp := lin.Fprop(q, input, nil, true)
		output := relu.Fprop(q, temp, nil, true)

		inGrad := inputGrad(t, q, output, yOneHot)
		temp2 := relu.Bprop(q, inGrad, nil)
		lin.Bprop(q, temp2, nil)

		expect := []float32{
			0, 0, 0, 0, 0, 0,
			0.009336092, 0.10726756, -0.06394093, -0.27729696, 0.36944908, 0.31641093,
			-0.8669185, -0.6640028, -0.3048705, -0.31535587, -1.0578532, -0.94342196,
			0.063568585, 0.052299034, -0.0022429964, 0.14734498, 0.09623108, 0.12843394}
		compareArray(t, q, "dW", dW, expect)
		expect = []float32{0, 0.096598804, -1.27191, 0.18208665}
		compareArray(t, q, "dB", dB, expect)

		q.Shutdown()
	}
}
