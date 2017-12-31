package main

import (
	"fmt"
	"github.com/jnb666/deepthought2/nnet"
)

func main() {
	conf := nnet.Config{
		DataSet:    "cifar10",
		Eta:        0.05,
		Lambda:     5,
		TrainRuns:  5,
		MaxEpoch:   100,
		TrainBatch: 64,
		TestBatch:  250,
		Shuffle:    true,
		UseGPU:     true,
		Normalise:  true,
		Distort:    true,
		WeightInit: nnet.GlorotUniform,
	}.AddLayers(
		nnet.Conv{Nfeats: 32, Size: 3, Pad: 1},
		nnet.Activation{Atype: "relu"},
		nnet.Conv{Nfeats: 32, Size: 3, Pad: 1},
		nnet.Activation{Atype: "relu"},
		nnet.MaxPool{Size: 2},
		nnet.Dropout{Ratio: 0.2},

		nnet.Conv{Nfeats: 64, Size: 3, Pad: 1},
		nnet.Activation{Atype: "relu"},
		nnet.Conv{Nfeats: 64, Size: 3, Pad: 1},
		nnet.Activation{Atype: "relu"},
		nnet.MaxPool{Size: 2},
		nnet.Dropout{Ratio: 0.3},

		nnet.Conv{Nfeats: 128, Size: 3, Pad: 1},
		nnet.Activation{Atype: "relu"},
		nnet.Conv{Nfeats: 128, Size: 3, Pad: 1},
		nnet.Activation{Atype: "relu"},
		nnet.MaxPool{Size: 2},
		nnet.Dropout{Ratio: 0.4},

		nnet.Flatten{},
		nnet.Linear{Nout: 10},
		nnet.Activation{Atype: "softmax"},
	)
	fmt.Println(conf)
	err := conf.Save("cifar10_6.conf")
	nnet.CheckErr(err)
}
