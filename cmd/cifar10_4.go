// cifar-10 4 layer net based on https://blog.plon.io/tutorials/cifar-10-classification-using-keras-tutorial/
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
		MaxEpoch:   100,
		MaxSeconds: 180,
		TrainBatch: 40,
		Shuffle:    true,
		Distort:    true,
		UseGPU:     true,
		Normalise:  true,
		FastConv:   true,
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
}
