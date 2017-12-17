package main

import (
	"fmt"
	"github.com/jnb666/deepthought2/nnet"
)

func main() {
	conf := nnet.Config{
		DataSet:       "mnist2",
		Eta:           0.03,
		Lambda:        0.1,
		NormalWeights: true,
		MaxEpoch:      30,
		TrainBatch:    10,
		TestBatch:     100,
		Shuffle:       true,
		UseGPU:        true,
		StopAfter:     2,
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
		nnet.LogRegression{},
	)
	fmt.Println(conf)
	err := conf.Save("mnist_cnn_deep2.conf")
	nnet.CheckErr(err)
}
