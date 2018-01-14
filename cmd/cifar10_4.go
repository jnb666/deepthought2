package main

import (
	"fmt"
	"github.com/jnb666/deepthought2/nnet"
)

func main() {
	conf := nnet.Config{
		DataSet:    "cifar10",
		Eta:        0.05,
		Lambda:     0.5,
		TrainRuns:  5,
		MaxEpoch:   50,
		TrainBatch: 40,
		Shuffle:    true,
		UseGPU:     true,
		Normalise:  true,
		WeightInit: nnet.GlorotUniform,
	}.AddLayers(
		nnet.Conv{Nfeats: 32, Size: 3, Pad: true},
		nnet.Activation{Atype: "relu"},
		nnet.Conv{Nfeats: 32, Size: 3},
		nnet.Activation{Atype: "relu"},
		nnet.Pool{Size: 2},
		nnet.Dropout{Ratio: 0.25},
		nnet.Conv{Nfeats: 64, Size: 3, Pad: true},
		nnet.Activation{Atype: "relu"},
		nnet.Conv{Nfeats: 64, Size: 3},
		nnet.Activation{Atype: "relu"},
		nnet.Pool{Size: 2},
		nnet.Dropout{Ratio: 0.25},
		nnet.Flatten{},
		nnet.Linear{Nout: 512},
		nnet.Activation{Atype: "relu"},
		nnet.Dropout{Ratio: 0.5},
		nnet.Linear{Nout: 10},
		nnet.Activation{Atype: "softmax"},
	)
	fmt.Println(conf)
	err := conf.Save("cifar10_4.conf")
	nnet.CheckErr(err)

	// with image distortion
	conf.Distort = true
	conf.MaxEpoch = 100
	err = conf.Save("cifar10_4d.conf")
	nnet.CheckErr(err)
}
