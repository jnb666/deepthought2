package main

import (
	"fmt"
	"github.com/jnb666/deepthought2/nnet"
)

func main() {
	data := nnet.Data{
		Classes: 1,
		Shape:   []int{2},
		Input:   []float32{0, 0, 0, 1, 1, 0, 1, 1},
		Labels:  []int32{0, 1, 1, 0},
	}
	err := data.Save("xor_train")
	nnet.CheckErr(err)

	conf := nnet.Config{
		DataSet:      "xor",
		Eta:          10,
		MaxEpoch:     500,
		LogEvery:     25,
		MinLoss:      0.05,
		FlattenInput: true,
		Threads:      1,
	}.AddLayers(
		nnet.Linear{Nout: 2},
		nnet.Activation{Atype: "sigmoid"},
		nnet.Linear{Nout: 1},
		nnet.Activation{Atype: "sigmoid"},
	)
	fmt.Println(conf)

	err = conf.SaveDefault("xor")
	nnet.CheckErr(err)
}
