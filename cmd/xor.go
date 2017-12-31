package main

import (
	"fmt"
	"github.com/jnb666/deepthought2/nnet"
)

type Data struct{}

func main() {
	data := nnet.NewData([]string{"0", "1"}, []int{2}, []int32{0, 1, 1, 0}, []float32{0, 0, 0, 1, 1, 0, 1, 1})
	err := nnet.SaveDataFile(data, "xor_train", false)
	nnet.CheckErr(err)

	conf := nnet.Config{
		DataSet:      "xor",
		Eta:          10,
		TrainRuns:    1,
		MaxEpoch:     500,
		LogEvery:     25,
		MinLoss:      0.05,
		FlattenInput: true,
		WeightInit:   nnet.RandomUniform,
	}.AddLayers(
		nnet.Linear{Nout: 2},
		nnet.Activation{Atype: "sigmoid"},
		nnet.Linear{Nout: 1},
		nnet.Activation{Atype: "sigmoid"},
	)
	fmt.Println(conf)

	err = conf.Save("xor.conf")
	nnet.CheckErr(err)
}
