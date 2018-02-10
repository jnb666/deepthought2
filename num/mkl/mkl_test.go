package mkl

import "testing"

type testCase struct {
	in, out, pad   int
	filter, stride int
	padding        bool
}

func TestOutSize(t *testing.T) {
	tests := []testCase{
		{in: 11, filter: 6, stride: 5, out: 2},
		{in: 13, filter: 6, stride: 5, out: 2},
		{in: 11, filter: 6, stride: 5, padding: true, out: 3, pad: 3},
		{in: 13, filter: 6, stride: 5, padding: true, out: 3, pad: 2},
		{in: 28, filter: 5, stride: 1, out: 24},
		{in: 28, filter: 5, stride: 1, padding: true, out: 28, pad: 2},
		{in: 32, filter: 1, stride: 2, out: 16},
		{in: 32, filter: 1, stride: 2, padding: true, out: 16},
	}
	for _, test := range tests {
		out, pad := getOutSize(test.in, test.filter, test.stride, test.padding)
		t.Logf("in=%d filter=%d stride=%d padding=%v => out=%d pad=%d", test.in, test.filter, test.stride, test.padding, out, pad)
		if out != test.out || pad != test.pad {
			t.Error("**ERROR**")
		}
	}
}
