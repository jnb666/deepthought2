package main

import (
	"fmt"
	"github.com/jnb666/deepthought2/nnet"
)

func main() {
	confBase := nnet.Config{
		DataSet:      "mnist",
		Eta:          0.1,
		Lambda:       3.0,
		FlattenInput: true,
		TrainRuns:    1,
		MaxEpoch:     20,
		TrainBatch:   10,
		TestBatch:    100,
		Shuffle:      true,
		WeightInit:   nnet.LecunNormal,
	}
	conf := confBase.AddLayers(
		nnet.Linear{Nout: 100},
		nnet.Activation{Atype: "relu"},
		nnet.Linear{Nout: 10},
		nnet.Activation{Atype: "softmax"},
	)
	fmt.Println(conf)
	err := conf.Save("mnist_mlp.conf")
	nnet.CheckErr(err)
}
