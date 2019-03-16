// Network in network: see https://arxiv.org/abs/1312.4400
package main

import (
	"fmt"

	"github.com/jnb666/deepthought2/nnet"
)

func conv(nfeat, size int) []nnet.ConfigLayer {
	return []nnet.ConfigLayer{
		nnet.Conv{Nfeats: nfeat, Size: size, Pad: true, NoBias: true},
		nnet.BatchNorm{},
		nnet.Activation{Atype: "relu"},
	}
}

func main() {
	conf := nnet.Config{
		DataSet:      "cifar10_2",
		Optimiser:    nnet.NesterovOpt,
		Eta:          0.01,
		EtaDecay:     0.2,
		EtaDecayStep: 25,
		Lambda:       50,
		Momentum:     0.9,
		MaxEpoch:     200,
		StopAfter:    2,
		ExtraEpochs:  4,
		ValidEMA:     20,
		TrainBatch:   125,
		Shuffle:      true,
		UseGPU:       true,
		Normalise:    true,
		Distort:      true,
		WeightInit:   nnet.GlorotUniform,
	}

	c := conf.AddLayers(conv(192, 5)...)
	c = c.AddLayers(conv(160, 1)...)
	c = c.AddLayers(conv(96, 1)...)
	c = c.AddLayers(nnet.Pool{Size: 3, Stride: 2, Pad: true})

	c = c.AddLayers(conv(192, 5)...)
	c = c.AddLayers(conv(192, 1)...)
	c = c.AddLayers(conv(192, 1)...)
	c = c.AddLayers(nnet.Pool{Size: 3, Stride: 2, Pad: true, Average: true})

	c = c.AddLayers(conv(192, 3)...)
	c = c.AddLayers(conv(192, 1)...)
	c = c.AddLayers(conv(192, 1)...)
	c = c.AddLayers(
		nnet.Pool{Size: 8, Average: true},
		nnet.Flatten{},
		nnet.Linear{Nout: 10},
		nnet.Activation{Atype: "softmax"},
	)

	fmt.Println(c)
	err := c.Save("cifar10_nin.conf")
	nnet.CheckErr(err)
}
