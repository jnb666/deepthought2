package num

import (
	"math/rand"
	"reflect"
	"testing"
)

var devices []Device

func init() {
	devices = []Device{
		NewDevice(false),
		NewDevice(true),
	}
}

func TestArray(t *testing.T) {
	for _, dev := range devices {
		xd := []float32{1, 1, 2, 2, 3, 3}
		q := dev.NewQueue()
		x := dev.NewArray(Float32, 6)
		if typ := x.Dtype; typ != Float32 {
			t.Error("dtype invalid: got", typ)
		}
		x = x.Reshape(2, 3)
		if dim := x.Dims; !reflect.DeepEqual(dim, []int{2, 3}) {
			t.Error("dims invalid: got", dim)
		}
		res := make([]float32, 6)
		q.Call(
			Write(x, xd),
			Read(x, res),
		).Finish()
		t.Logf("x=\n%s", x.String(q))
		if !reflect.DeepEqual(res, xd) {
			t.Error("got", res, "expect", xd)
		}
		expect := []float32{9, 6, 8, 5, 7, 4}
		q.Call(
			Fill(x, 42),
			WriteCol(x, 0, []float32{9, 6}),
			WriteCol(x, 1, []float32{8, 5}),
			WriteCol(x, 2, []float32{7, 4}),
			Read(x, res),
		).Finish()
		t.Logf("writecol x=\n%s", x.String(q))
		if !reflect.DeepEqual(res, expect) {
			t.Error("got", res, "expect", expect)
		}
		q.Shutdown()
	}
}

func TestCopy(t *testing.T) {
	for _, dev := range devices {
		q := dev.NewQueue()
		x := dev.NewArray(Float32, 2, 3)
		// tile columns
		y := dev.NewArray(Float32, 2, 1)
		res := make([]float32, 6)
		q.Call(
			Write(y, []float32{1, 2}),
			Copy(y, x),
			Read(x, res),
		).Finish()
		t.Logf("tile cols x=\n%s", x.String(q))
		expect := []float32{1, 2, 1, 2, 1, 2}
		if !reflect.DeepEqual(res, expect) {
			t.Error("got", res, "expect", expect)
		}
		// tile rows
		y = dev.NewArray(Float32, 3)
		q.Call(
			Write(y, []float32{3, 2, 1}),
			Copy(y, x),
			Read(x, res),
		).Finish()
		t.Logf("tile rows x=\n%s", x.String(q))
		expect = []float32{3, 3, 2, 2, 1, 1}
		if !reflect.DeepEqual(res, expect) {
			t.Error("got", res, "expect", expect)
		}
		q.Shutdown()
	}
}

func TestOnehot(t *testing.T) {
	for _, dev := range devices {
		q := dev.NewQueue()
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
		q.Shutdown()
	}
}

func TestTranspose(t *testing.T) {
	for _, dev := range devices {
		q := dev.NewQueue()
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
		q.Shutdown()
	}
}

func TestScale(t *testing.T) {
	for _, dev := range devices {
		q := dev.NewQueue()
		x := dev.NewArray(Float32, 2, 3)
		res := make([]float32, 6)
		q.Call(
			Write(x, []float32{1, 2, 3, 4, 5, 6}),
			Scale(5, x),
			Read(x, res),
		).Finish()
		t.Logf("scale x=\n%s", x.String(q))
		expect := []float32{5, 10, 15, 20, 25, 30}
		if !reflect.DeepEqual(res, expect) {
			t.Error("got", res, "expect", expect)
		}
		q.Shutdown()
	}
}

func TestAxpy(t *testing.T) {
	for _, dev := range devices {
		q := dev.NewQueue()
		x := dev.NewArray(Float32, 2, 3)
		y := dev.NewArray(Float32, 2, 3)
		res := make([]float32, 6)
		q.Call(
			Write(x, []float32{1, 1, 2, 2, 3, 3}),
			Write(y, []float32{0.5, 0.5, 0.5, 0.5, 0.5, 0.5}),
			Axpy(2, x, y),
			Read(y, res),
		).Finish()
		t.Logf("axpy y=\n%s", y.String(q))
		expect := []float32{2.5, 2.5, 4.5, 4.5, 6.5, 6.5}
		if !reflect.DeepEqual(res, expect) {
			t.Error("got", res, "expect", expect)
		}
		q.Shutdown()
	}
}

func TestSum(t *testing.T) {
	for _, dev := range devices {
		q := dev.NewQueue()
		x := dev.NewArray(Float32, 2, 3)
		sum := dev.NewArray(Float32)
		res := make([]float32, 1)
		// scalar sum
		q.Call(
			Write(x, []float32{1, 2, 3, 4, 5, 6}),
			Sum(x, sum),
			Read(sum, res),
		).Finish()
		t.Log("scalar sum", res)
		if res[0] != 21 {
			t.Error("got", res[0], "expect", 3.5)
		}
		// sum for each column
		sum = dev.NewArray(Float32, 3)
		res = make([]float32, 3)
		ones := dev.NewArray(Float32, 2)
		q.Call(
			Fill(ones, 1),
			Gemv(x, ones, sum, Trans),
			Read(sum, res),
		).Finish()
		t.Log("column sum", res)
		expect := []float32{3, 7, 11}
		if !reflect.DeepEqual(res, expect) {
			t.Error("got", res, "expect", expect)
		}
		q.Shutdown()
	}
}

func TestGemm(t *testing.T) {
	for _, dev := range devices {
		q := dev.NewQueue()
		x := dev.NewArray(Float32, 2, 3)
		y := dev.NewArray(Float32, 3, 2)
		z := dev.NewArray(Float32, 2, 2)
		q.Call(
			Write(x, []float32{1, 4, 2, 5, 3, 6}),
		)
		res := make([]float32, 4)
		for _, trans := range []TransType{NoTrans, Trans} {
			if trans == Trans {
				y = y.Reshape(2, 3)
				q.Call(Write(y, []float32{7, 8, 9, 10, 11, 12}))
			} else {
				q.Call(Write(y, []float32{7, 9, 11, 8, 10, 12}))
			}
			q.Call(
				Gemm(x, y, z, NoTrans, trans, false),
				Read(z, res),
			).Finish()
			t.Logf("gemm z=\n%s", z.String(q))
			expect := []float32{58, 139, 64, 154}
			if !reflect.DeepEqual(res, expect) {
				t.Error("got", res, "expect", expect)
			}
		}
		q.Shutdown()
	}
}

func TestMul(t *testing.T) {
	for _, dev := range devices {
		q := dev.NewQueue()
		x := dev.NewArray(Float32, 2, 3)
		y := dev.NewArrayLike(x)
		z := dev.NewArrayLike(x)
		res := make([]float32, 6)
		q.Call(
			Write(x, []float32{1, 2, 3, 4, 5, 6}),
			Write(y, []float32{0.5, 0.1, 0.1, 0.1, 0.1, 0.5}),
			Mul(x, y, z),
			Read(z, res),
		).Finish()
		t.Logf("mul z=\n%s", z.String(q))
		expect := []float32{0.5, 0.2, 0.3, 0.4, 0.5, 3}
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

func benchGemm(b *testing.B, dev Device) {
	size := 250
	q := dev.NewQueue()
	x := dev.NewArray(Float32, size, size)
	y := dev.NewArray(Float32, size, size)
	z := dev.NewArray(Float32, size, size)
	q.Call(
		Write(x, randSlice(size*size)),
		Write(y, randSlice(size*size)),
	).Finish()
	for i := 0; i < b.N; i++ {
		q.Call(Gemm(x, y, z, NoTrans, NoTrans, false)).Finish()
	}
}

func BenchmarkGemmCPU(b *testing.B) {
	benchGemm(b, devices[0])
}

func BenchmarkGemmGPU(b *testing.B) {
	benchGemm(b, devices[1])
}
