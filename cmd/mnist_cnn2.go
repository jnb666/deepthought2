package main

import (
	"fmt"
	"github.com/jnb666/deepthought2/nnet"
)

func main() {
	conf := nnet.Config{
		DataSet:    "mnist",
		Eta:        0.16,
		Lambda:     0.1,
		TrainRuns:  1,
		MaxEpoch:   20,
		TrainBatch: 64,
		TestBatch:  250,
		Shuffle:    true,
		UseGPU:     true,
		WeightInit: nnet.LecunNormal,
	}.AddLayers(
		nnet.Conv{Nfeats: 20, Size: 5},
		nnet.Activation{Atype: "relu"},
		nnet.MaxPool{Size: 2},
		nnet.Conv{Nfeats: 40, Size: 5},
		nnet.Activation{Atype: "relu"},
		nnet.MaxPool{Size: 2},
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
	conf.StopAfter = 1
	conf.ExtraEpochs = 2
	fmt.Println(conf)
	err = conf.Save("mnist_cnn2d.conf")
	nnet.CheckErr(err)
}
