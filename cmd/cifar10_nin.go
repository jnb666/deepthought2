package main

import (
	"fmt"
	"github.com/jnb666/deepthought2/nnet"
)

func conv(nfeat, size int) []nnet.ConfigLayer {
	return []nnet.ConfigLayer{
		nnet.Conv{Nfeats: nfeat, Size: size, Pad: size > 1, NoBias: true},
		nnet.BatchNorm{},
		nnet.Activation{Atype: "relu"},
	}
}

func main() {
	conf := nnet.Config{
		DataSet:     "cifar10",
		Eta:         0.1,
		Lambda:      250,
		TrainRuns:   5,
		MaxEpoch:    100,
		TrainBatch:  100,
		Shuffle:     true,
		UseGPU:      true,
		Normalise:   true,
		Distort:     true,
		WeightInit:  nnet.HeNormal,
		StopAfter:   2,
		ExtraEpochs: 4,
	}

	conf = conf.AddLayers(conv(192, 5)...)
	conf = conf.AddLayers(conv(160, 1)...)
	conf = conf.AddLayers(conv(96, 1)...)
	conf = conf.AddLayers(nnet.Pool{Size: 3, Stride: 2, Pad: true})

	conf = conf.AddLayers(conv(192, 5)...)
	conf = conf.AddLayers(conv(192, 1)...)
	conf = conf.AddLayers(conv(192, 1)...)
	conf = conf.AddLayers(nnet.Pool{Size: 3, Stride: 2, Pad: true, Average: true})

	conf = conf.AddLayers(conv(192, 3)...)
	conf = conf.AddLayers(conv(192, 1)...)
	conf = conf.AddLayers(conv(192, 1)...)
	conf = conf.AddLayers(
		nnet.Pool{Size: 8, Average: true},
		nnet.Flatten{},
		nnet.Linear{Nout: 10},
		nnet.Activation{Atype: "softmax"},
	)

	fmt.Println(conf)
	err := conf.Save("cifar10_nin.conf")
	nnet.CheckErr(err)
}
