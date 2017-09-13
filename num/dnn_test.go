package num

import (
	"github.com/jnb666/deepthought2/num/dnn"
	"math/rand"
	"testing"
	"time"
)

const (
	randomise = false
	batch     = 5
	nIn       = 6
	nOut      = 4
	eps       = 1e-6
)

var dev = NewCPUDevice()

func randArray(size int, min, max float32) []float32 {
	v := make([]float32, size)
	for i := range v {
		v[i] = min + rand.Float32()*(max-min)
	}
	return v
}

func getInputs(t *testing.T, q Queue) (input, W, B Array) {
	if randomise {
		rand.Seed(time.Now().UTC().UnixNano())
	} else {
		rand.Seed(42)
	}
	input = dev.NewArray(Float32, batch, nIn)
	W = dev.NewArray(Float32, nIn, nOut)
	B = dev.NewArray(Float32, nOut)
	weights := randArray(nIn*nOut, -0.5, 0.5)
	bias := randArray(nOut, 0.1, 0.2)
	inData := randArray(batch*nIn, 0, 1)
	q.Call(
		Write(W, weights),
		Write(B, bias),
		Write(input, inData),
	)
	t.Logf("== weights ==\n%s %s", W.String(q), B.String(q))
	t.Logf("== input ==\n%s", input.String(q))
	return
}

func getOutput(q Queue) Array {
	data := make([]float32, nOut*batch)
	for row := 0; row < batch; row++ {
		data[row+rand.Intn(nOut)*batch] = 1
	}
	yOneHot := dev.NewArray(Float32, batch, nOut)
	q.Call(Write(yOneHot, data))
	return yOneHot
}

func inputGrad(t *testing.T, q Queue, output, yOneHot Array) Array {
	inGrad := dev.NewArrayLike(output)
	q.Call(
		Copy(inGrad, output),
		Axpy(-1, yOneHot, inGrad),
	)
	t.Logf("== input grad ==\n%s", inGrad.String(q))
	return inGrad
}

func setupNetwork() (linear, relu dnn.Layer) {
	linear = dev.LinearLayer(batch, nIn, nOut)
	relu = dev.ReluLayer(linear)
	return
}

func fpropBLAS(q Queue, input, W, B Array) (output, temp Array) {
	output = dev.NewArray(Float32, batch, nOut)
	temp = dev.NewArrayLike(output)
	q.Call(
		Copy(temp, B),
		Gemm(1, 1, input, W, temp, NoTrans, NoTrans),
		Relu(temp, output),
	)
	return
}

func bpropBLAS(q Queue, inGrad, input, input1, W Array) (dSrc, dW, dB Array) {
	dSrc = dev.NewArray(Float32, batch, nIn)
	dW = dev.NewArray(Float32, nIn, nOut)
	dB = dev.NewArray(Float32, nOut)
	grad := dev.NewArray(Float32, batch, nOut)
	scale := 1 / float32(batch)
	ones := dev.NewArray(Float32, batch)
	q.Call(
		ReluD(input1, inGrad, grad),
		Fill(ones, 1),
		Gemv(scale, 0, grad, ones, dB, Trans),
		Gemm(scale, 0, input, grad, dW, Trans, NoTrans),
		Gemm(1, 0, grad, W, dSrc, NoTrans, Trans),
	)
	return
}

func fpropDNN(t *testing.T, q Queue, linear, relu dnn.Layer, input, W, B Array) Array {
	output := dev.NewArray(Float32, batch, nOut)
	ArrayIn(linear, dnn.Filter, W, NoTrans)
	ArrayIn(linear, dnn.Bias, B, NoTrans)
	ArrayIn(linear, dnn.Src, input, Trans)
	ArrayOut(relu, dnn.Dst, output, Trans)
	t.Log("== FPROP ==\n", linear, "\n", relu)
	q.Call(
		Set(linear, dnn.Filter, W),
		Set(linear, dnn.Bias, B),
		Set(linear, dnn.Src, input),
		Fprop(linear),
		Fprop(relu),
		Get(relu, dnn.Dst, output),
	)
	return output
}

func bpropDNN(t *testing.T, q Queue, linear, relu dnn.Layer, inGrad Array) (dSrc, dW, dB Array) {
	dSrc = dev.NewArray(Float32, batch, nIn)
	dW = dev.NewArray(Float32, nIn, nOut)
	dB = dev.NewArray(Float32, nOut)
	ArrayOut(linear, dnn.DiffFilter, dW, NoTrans)
	ArrayOut(linear, dnn.DiffBias, dB, NoTrans)
	ArrayOut(linear, dnn.DiffSrc, dSrc, Trans)
	ArrayIn(relu, dnn.DiffDst, inGrad, Trans)
	t.Log("== BPROP ==\n", linear, "\n", relu)
	scale := 1 / float32(batch)
	q.Call(
		Set(relu, dnn.DiffDst, inGrad),
		BpropData(relu),
		BpropData(linear),
		BpropFilter(linear),
		BpropBias(linear),
		Get(linear, dnn.DiffSrc, dSrc),
		Get(linear, dnn.DiffFilter, dW),
		Get(linear, dnn.DiffBias, dB),
		Scale(scale, dW),
		Scale(scale, dB),
	)
	return
}

func compareArrays(t *testing.T, q Queue, title string, A, B Array) {
	t.Logf("== %s DNN ==\n%s", title, A.String(q))
	t.Logf("== %s BLAS ==\n%s", title, B.String(q))
	arrA := make([]float32, Prod(A.Dims()))
	arrB := make([]float32, Prod(B.Dims()))
	q.Call(
		Read(A, arrA),
		Read(B, arrB),
	).Finish()
	if len(arrA) != len(arrB) {
		t.Fatal(title, "length mismatch!")
	}
	for i := range arrA {
		if abs(arrA[i]-arrB[i]) > eps {
			t.Error(title, "mismatch!")
			return
		}
	}
}

func TestDNNFprop(t *testing.T) {
	q := dev.NewQueue(1)
	linear, relu := setupNetwork()
	input, W, B := getInputs(t, q)
	outBLAS, _ := fpropBLAS(q, input, W, B)
	outDNN := fpropDNN(t, q, linear, relu, input, W, B)
	compareArrays(t, q, "output", outDNN, outBLAS)
	q.Shutdown()
}

func TestDNNBprop(t *testing.T) {
	q := dev.NewQueue(1)
	linear, relu := setupNetwork()
	input, W, B := getInputs(t, q)
	yOneHot := getOutput(q)

	output := fpropDNN(t, q, linear, relu, input, W, B)
	inGrad := inputGrad(t, q, output, yOneHot)
	dSrcDNN, dWDNN, dBDNN := bpropDNN(t, q, linear, relu, inGrad)

	output, temp := fpropBLAS(q, input, W, B)
	inGrad = inputGrad(t, q, output, yOneHot)
	dSrcBLAS, dWBLAS, dBBLAS := bpropBLAS(q, inGrad, input, temp, W)

	compareArrays(t, q, "dSrc", dSrcDNN, dSrcBLAS)
	compareArrays(t, q, "dW", dWDNN, dWBLAS)
	compareArrays(t, q, "dB", dBDNN, dBBLAS)

	q.Shutdown()
}
