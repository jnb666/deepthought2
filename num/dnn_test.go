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
	input = dev.NewArray(Float32, nIn, batch)
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

func setupNetwork(W, B Array) (linear, relu dnn.Layer, dW, dB Array) {
	dW = dev.NewArrayLike(W)
	dB = dev.NewArrayLike(B)
	linear = dev.LinearLayer(batch, nIn, nOut, W, B, dW, dB)
	relu = dev.ReluLayer(linear)
	relu.SetData(dnn.Src, linear.Data(dnn.Dst))
	linear.SetData(dnn.DiffDst, relu.Data(dnn.DiffSrc))
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
func fpropBLAS(q Queue, input, W, B Array) (output, output1 Array) {
	output = dev.NewArray(Float32, nOut, batch)
	output1 = dev.NewArrayLike(output)
	temp := dev.NewArray(Float32, batch, nOut)
	q.Call(
		Copy(temp, B),
		Gemm(1, 1, input, W, temp, Trans, NoTrans),
		Transpose(temp, output1),
		Relu(output1, output),
	)
	return
}

func fpropDNN(t *testing.T, q Queue, linear, relu dnn.Layer, input Array) Array {
	linear.SetData(dnn.Src, input.Data())
	output := dev.NewArrayFrom(relu, dnn.Dst)
	t.Logf("== FPROP ==\n%s%s", linear, relu)
	q.Call(
		Fprop(linear),
		Fprop(relu),
	)
	return output
}

func TestDNNFprop(t *testing.T) {
	q := dev.NewQueue(1)
	input, W, B := getInputs(t, q)
	linear, relu, _, _ := setupNetwork(W, B)
	dst := fpropDNN(t, q, linear, relu, input)
	dstBLAS, _ := fpropBLAS(q, input, W, B)
	compareArrays(t, q, "output", dst, dstBLAS)
	q.Shutdown()
}

func getOutput(q Queue) Array {
	data := make([]float32, nOut*batch)
	for row := 0; row < batch; row++ {
		data[row+rand.Intn(nOut)*batch] = 1
	}
	yOneHot := dev.NewArray(Float32, nOut, batch)
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

func bpropBLAS(q Queue, inGrad, input, input1, W Array) (dSrc, dW, dB Array) {
	dSrc = dev.NewArray(Float32, nIn, batch)
	dSrcT := dev.NewArray(Float32, batch, nIn)
	dW = dev.NewArray(Float32, nIn, nOut)
	dB = dev.NewArray(Float32, nOut)
	grad := dev.NewArray(Float32, nOut, batch)
	scale := 1 / float32(batch)
	ones := dev.NewArray(Float32, batch)
	q.Call(
		ReluD(input1, inGrad, grad),
		Fill(ones, 1),
		Gemv(scale, 0, grad, ones, dB, NoTrans),
		Gemm(scale, 0, input, grad, dW, NoTrans, Trans),
		Gemm(1, 0, grad, W, dSrcT, Trans, Trans),
		Transpose(dSrcT, dSrc),
	)
	return
}

func bpropDNN(t *testing.T, q Queue, linear, relu dnn.Layer, inGrad, dW, dB Array) Array {
	relu.SetData(dnn.DiffDst, inGrad.Data())
	dSrc := dev.NewArrayFrom(linear, dnn.DiffSrc)
	scale := 1 / float32(batch)
	t.Logf("== BPROP ==\n%s%s", linear, relu)
	q.Call(
		BpropData(relu),
		BpropData(linear),
		BpropFilter(linear),
		BpropBias(linear),
		Scale(scale, dW),
		Scale(scale, dB),
	)
	return dSrc
}

func TestDNNBprop(t *testing.T) {
	q := dev.NewQueue(1)
	input, W, B := getInputs(t, q)
	linear, relu, dW, dB := setupNetwork(W, B)
	yOneHot := getOutput(q)

	output := fpropDNN(t, q, linear, relu, input)
	inGrad := inputGrad(t, q, output, yOneHot)
	dSrc := bpropDNN(t, q, linear, relu, inGrad, dW, dB)

	output, temp := fpropBLAS(q, input, W, B)
	inGrad = inputGrad(t, q, output, yOneHot)
	dSrcBLAS, dWBLAS, dBBLAS := bpropBLAS(q, inGrad, input, temp, W)

	compareArrays(t, q, "dSrc", dSrc, dSrcBLAS)
	compareArrays(t, q, "dW", dW, dWBLAS)
	compareArrays(t, q, "dB", dB, dBBLAS)

	q.Shutdown()
}
