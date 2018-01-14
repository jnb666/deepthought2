package mkl

import "testing"

type testCase struct {
	in, out, pad   int
	filter, stride int
	padding, err   bool
}

func TestOutSize(t *testing.T) {
	tests := []testCase{
		{in: 11, filter: 6, stride: 5, out: 2},
		{in: 13, filter: 6, stride: 5, out: 2, err: true},
		{in: 11, filter: 6, stride: 5, padding: true, out: 2},
		{in: 13, filter: 6, stride: 5, padding: true, out: 3, pad: 2},
		{in: 28, filter: 5, stride: 1, out: 24},
		{in: 28, filter: 5, stride: 1, padding: true, out: 28, pad: 2},
	}
	for _, test := range tests {
		out, pad, err := getOutSize(test.in, test.filter, test.stride, test.padding)
		t.Logf("in=%d filter=%d stride=%d padding=%v => out=%d pad=%d err=%v", test.in, test.filter, test.stride, test.padding, out, pad, err)
		if out != test.out || pad != test.pad || (err != nil) != test.err {
			t.Error("**ERROR**")
		}
	}
}
