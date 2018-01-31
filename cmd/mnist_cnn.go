// mnist conv net based on http://www.imm.dtu.dk/~abll/blog/simple_cnn/
package main

import (
	"fmt"
	"github.com/jnb666/deepthought2/nnet"
)

func main() {
	conf := nnet.Config{
		DataSet:    "mnist",
		Eta:        0.1,
		TrainRuns:  1,
		MaxEpoch:   10,
		TrainBatch: 10,
		TestBatch:  100,
		Shuffle:    true,
		UseGPU:     true,
		WeightInit: nnet.LecunNormal,
	}.AddLayers(
		nnet.Conv{Nfeats: 20, Size: 5, Pad: true},
		nnet.Activation{Atype: "relu"},
		nnet.Pool{Size: 2},
		nnet.Flatten{},
		nnet.Linear{Nout: 100},
		nnet.Activation{Atype: "relu"},
		nnet.Linear{Nout: 10},
		nnet.Activation{Atype: "softmax"},
	)
	fmt.Println(conf)
	err := conf.Save("mnist_cnn.conf")
	nnet.CheckErr(err)
}
