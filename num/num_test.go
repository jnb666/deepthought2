package num

import (
	"math/rand"
	"reflect"
	"testing"
)

func TestArray(t *testing.T) {
	xd := []float32{1, 1, 2, 2, 3, 3}
	dev := NewCPUDevice()
	q := dev.NewQueue(1)
	x := dev.NewArray(Float32, 6)
	if typ := x.Dtype(); typ != Float32 {
		t.Error("dtype invalid: got", typ)
	}
	x = x.Reshape(2, 3)
	if dim := x.Dims(); !reflect.DeepEqual(dim, []int{2, 3}) {
		t.Error("dims invalid: got", dim)
	}
	res := make([]float32, 6)
	q.Call(
		Write(x, xd),
		Read(x, res),
	).Finish()
	if !reflect.DeepEqual(res, xd) {
		t.Error("got", res, "expect", xd)
	}
	expect := []float32{9, 6, 8, 5, 7, 4}
	q.Call(
		Fill(x, 0),
		WriteCol(x, 0, []float32{9, 6}),
		WriteCol(x, 1, []float32{8, 5}),
		WriteCol(x, 2, []float32{7, 4}),
		Read(x, res),
	).Finish()
	if !reflect.DeepEqual(res, expect) {
		t.Error("got", res, "expect", expect)
	}
	q.Call(
		Fill(x, 0),
		WriteRow(x, 0, []float32{9, 8, 7}),
		WriteRow(x, 1, []float32{6, 5, 4}),
		Read(x, res),
	).Finish()
	if !reflect.DeepEqual(res, expect) {
		t.Error("got", res, "expect", expect)
	}
}

func TestCopy(t *testing.T) {
	dev := NewCPUDevice()
	q := dev.NewQueue(1)
	x := dev.NewArray(Float32, 2, 3)
	// tile columns
	y := dev.NewArray(Float32, 2, 1)
	res := make([]float32, 6)
	q.Call(
		Write(y, []float32{1, 2}),
		Copy(x, y),
		Read(x, res),
	).Finish()
	expect := []float32{1, 2, 1, 2, 1, 2}
	if !reflect.DeepEqual(res, expect) {
		t.Error("got", res, "expect", expect)
	}
	// tile rows
	y = dev.NewArray(Float32, 3)
	q.Call(
		Write(y, []float32{3, 2, 1}),
		Copy(x, y),
		Read(x, res),
	).Finish()
	expect = []float32{3, 3, 2, 2, 1, 1}
	if !reflect.DeepEqual(res, expect) {
		t.Error("got", res, "expect", expect)
	}
}

func TestOnehot(t *testing.T) {
	dev := NewCPUDevice()
	q := dev.NewQueue(1)
	y := dev.NewArray(Int32, 4)
	y1h := dev.NewArray(Float32, 3, 4)
	res := make([]float32, 12)
	vec := []int32{2, 1, 0, 2}
	q.Call(
		Write(y, vec),
		Onehot(y, y1h, 3),
		Read(y1h, res),
	).Finish()
	t.Logf("y1hot %s\n%s", y.String(q), y1h.String(q))
	expect := []float32{0, 0, 1, 0, 1, 0, 1, 0, 0, 0, 0, 1}
	if !reflect.DeepEqual(res, expect) {
		t.Error("got", res, "expect", expect)
	}
	res2 := make([]int32, 4)
	q.Call(
		Unhot(y1h, y),
		Read(y, res2),
	).Finish()
	if !reflect.DeepEqual(res2, vec) {
		t.Error("got", res2, "expect", vec)
	}
}

func TestTranspose(t *testing.T) {
	dev := NewCPUDevice()
	q := dev.NewQueue(1)
	x := dev.NewArray(Float32, 2, 3)
	y := dev.NewArray(Float32, 3, 2)
	res1 := make([]float32, 6)
	q.Call(
		Write(x, []float32{1, 1, 2, 2, 3, 3}),
		Transpose(x, y),
		Read(y, res1),
	).Finish()
	t.Logf("x\n%v", x.String(q))
	t.Logf("y\n%v", y.String(q))
	xT := []float32{1, 2, 3, 1, 2, 3}
	if !reflect.DeepEqual(res1, xT) {
		t.Error("got", res1, "expect", xT)
	}
}

func TestAxpy(t *testing.T) {
	dev := NewCPUDevice()
	q := dev.NewQueue(1)
	x := dev.NewArray(Float32, 2, 3)
	y := dev.NewArray(Float32, 2, 3)
	res := make([]float32, 6)
	q.Call(
		Write(x, []float32{1, 1, 2, 2, 3, 3}),
		Write(y, []float32{0.5, 0.5, 0.5, 0.5, 0.5, 0.5}),
		Axpy(2, x, y),
		Read(y, res),
	).Finish()
	expect := []float32{2.5, 2.5, 4.5, 4.5, 6.5, 6.5}
	if !reflect.DeepEqual(res, expect) {
		t.Error("got", res, "expect", expect)
	}
}

func TestSum(t *testing.T) {
	dev := NewCPUDevice()
	q := dev.NewQueue(1)
	x := dev.NewArray(Float32, 2, 3)
	sum := dev.NewArray(Float32)
	res := make([]float32, 1)
	// scalar sum
	q.Call(
		Write(x, []float32{1, 2, 3, 4, 5, 6}),
		Sum(x, sum, 1.0/6.0),
		Read(sum, res),
	).Finish()
	if res[0] != 3.5 {
		t.Error("got", res[0], "expect", 3.5)
	}
	// sum for each column
	sum = dev.NewArray(Float32, 3)
	res = make([]float32, 3)
	ones := dev.NewArray(Float32, 2)
	q.Call(
		Fill(ones, 1),
		Gemv(1, 0, x, ones, sum, Trans),
		Read(sum, res),
	).Finish()
	expect := []float32{3, 7, 11}
	if !reflect.DeepEqual(res, expect) {
		t.Error("got", res, "expect", expect)
	}
}

func TestGemm(t *testing.T) {
	dev := NewCPUDevice()
	q := dev.NewQueue(1)
	x := dev.NewArray(Float32, 2, 3)
	y := dev.NewArray(Float32, 3, 2)
	z := dev.NewArray(Float32, 2, 2)
	q.Call(Write(x, []float32{1, 4, 2, 5, 3, 6}))
	res := make([]float32, 4)
	for _, trans := range []TransType{NoTrans, Trans} {
		if trans == Trans {
			y = y.Reshape(2, 3)
			q.Call(Write(y, []float32{7, 8, 9, 10, 11, 12}))
		} else {
			q.Call(Write(y, []float32{7, 9, 11, 8, 10, 12}))
		}
		q.Call(
			Gemm(1, 0, x, y, z, NoTrans, trans),
			Read(z, res),
		).Finish()
		expect := []float32{58, 139, 64, 154}
		if !reflect.DeepEqual(res, expect) {
			t.Error("got", res, "expect", expect)
		}
	}
}

func randSlice(n int) []float32 {
	res := make([]float32, n)
	for i := range res {
		res[i] = float32(rand.Intn(20))
	}
	return res
}

func BenchmarkGemm(b *testing.B) {
	size := 100
	dev := NewCPUDevice()
	q := dev.NewQueue(4)
	x := dev.NewArray(Float32, size, size)
	y := dev.NewArray(Float32, size, size)
	z := dev.NewArray(Float32, size, size)
	q.Call(
		Write(x, randSlice(size*size)),
		Write(y, randSlice(size*size)),
	).Finish()
	for i := 0; i < b.N; i++ {
		q.Call(Gemm(1, 0, x, y, z, NoTrans, NoTrans)).Finish()
	}
}
