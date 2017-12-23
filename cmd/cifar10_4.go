package main

import (
	"fmt"
	"github.com/jnb666/deepthought2/nnet"
)

func main() {
	conf := nnet.Config{
		DataSet:       "cifar10",
		Eta:           0.1,
		Lambda:        0.5,
		Bias:          0.01,
		NormalWeights: true,
		TrainRuns:     10,
		MaxEpoch:      50,
		StopAfter:     2,
		ExtraEpochs:   3,
		TrainBatch:    32,
		TestBatch:     100,
		Shuffle:       true,
		UseGPU:        true,
	}.AddLayers(
		nnet.Conv{Nfeats: 32, Size: 3, Pad: 1},
		nnet.Activation{Atype: "relu"},
		nnet.Conv{Nfeats: 32, Size: 3},
		nnet.Activation{Atype: "relu"},
		nnet.MaxPool{Size: 2},
		nnet.Dropout{Ratio: 0.25},
		nnet.Conv{Nfeats: 64, Size: 3, Pad: 1},
		nnet.Activation{Atype: "relu"},
		nnet.Conv{Nfeats: 64, Size: 3},
		nnet.Activation{Atype: "relu"},
		nnet.MaxPool{Size: 2},
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
}
