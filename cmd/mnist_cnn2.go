// mnist conv net based on http://neuralnetworksanddeeplearning.com/chap6.html
package main

import (
	"fmt"
	"github.com/jnb666/deepthought2/nnet"
)

func main() {
	conf := nnet.Config{
		DataSet:    "mnist",
		Eta:        0.03,
		Lambda:     0.1,
		TrainRuns:  1,
		MaxEpoch:   20,
		TrainBatch: 10,
		TestBatch:  100,
		Shuffle:    true,
		UseGPU:     true,
		WeightInit: nnet.LecunNormal,
	}.AddLayers(
		nnet.Conv{Nfeats: 20, Size: 5},
		nnet.Activation{Atype: "relu"},
		nnet.Pool{Size: 2},
		nnet.Conv{Nfeats: 40, Size: 5},
		nnet.Activation{Atype: "relu"},
		nnet.Pool{Size: 2},
		nnet.Flatten{},
		nnet.Linear{Nout: 100},
		nnet.Activation{Atype: "relu"},
		nnet.Linear{Nout: 10},
		nnet.Activation{Atype: "softmax"},
	)
	fmt.Println(conf)
	err := conf.Save("mnist_cnn2.conf")
	nnet.CheckErr(err)

	// with image distortion
	conf.DataSet = "mnist2"
	conf.Distort = true
	conf.MaxEpoch = 40
	conf.ValidEMA = 10
	conf.StopAfter = 2
	conf.ExtraEpochs = 3
	conf.FastConv = true
	fmt.Println(conf)
	err = conf.Save("mnist_cnn2d.conf")
	nnet.CheckErr(err)
}
