package main

import (
	"fmt"
	"github.com/jnb666/deepthought2/nnet"
)

func main() {
	confBase := nnet.Config{
		DataSet:       "mnist",
		Eta:           0.1,
		Lambda:        3.0,
		NormalWeights: true,
		FlattenInput:  true,
		MaxEpoch:      20,
		TrainBatch:    10,
		TestBatch:     100,
		Threads:       4,
		Shuffle:       true,
	}
	conf := confBase.AddLayers(
		nnet.Linear{Nout: 100},
		nnet.Activation{Atype: "relu"},
		nnet.Linear{Nout: 10},
		nnet.LogRegression{},
	)
	fmt.Println(conf)
	err := conf.SaveDefault("mnist_mlp")
	nnet.CheckErr(err)

	conf = confBase.AddLayers(
		nnet.Linear{Nout: 100, DNN: true},
		nnet.Activation{Atype: "relu", DNN: true},
		nnet.Linear{Nout: 10, DNN: true},
		nnet.LogRegression{},
	)
	err = conf.SaveDefault("mnist_mlp_dnn")
	nnet.CheckErr(err)
}
