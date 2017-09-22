package main

import (
	"fmt"
	"github.com/jnb666/deepthought2/nnet"
)

func main() {
	conf := nnet.Config{
		DataSet:       "mnist",
		Eta:           0.03,
		Lambda:        0.1,
		NormalWeights: true,
		MaxEpoch:      10,
		TrainBatch:    10,
		TestBatch:     100,
		Threads:       4,
		Shuffle:       true,
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
	err := conf.SaveDefault("mnist_cnn_deep")
	nnet.CheckErr(err)
}
