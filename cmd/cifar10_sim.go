// Using simple architectures to outperform deeper and more complex architectures: see https://arxiv.org/pdf/1608.06037.pdf
package main

import (
	"fmt"
	"github.com/jnb666/deepthought2/nnet"
)

func block1(nfeat, size int) []nnet.ConfigLayer {
	return []nnet.ConfigLayer{
		nnet.Conv{Nfeats: nfeat, Size: size, Pad: true, NoBias: true},
		nnet.BatchNorm{AvgFactor: 0.05},
		nnet.Activation{Atype: "relu"},
		nnet.Dropout{Ratio: 0.2},
	}
}

func block2(nfeat, size int) []nnet.ConfigLayer {
	return []nnet.ConfigLayer{
		nnet.Conv{Nfeats: nfeat, Size: size, Pad: true, NoBias: true},
		nnet.BatchNorm{AvgFactor: 0.05},
		nnet.Activation{Atype: "relu"},
		nnet.Pool{Size: 2},
		nnet.Dropout{Ratio: 0.2},
	}
}

func main() {
	conf := nnet.Config{
		DataSet:      "cifar10_2",
		Eta:          0.01,
		EtaDecay:     0.2,
		EtaDecayStep: 30,
		Lambda:       50,
		Momentum:     0.9,
		Nesterov:     true,
		MaxEpoch:     200,
		StopAfter:    3,
		ExtraEpochs:  5,
		ValidEMA:     20,
		TrainBatch:   125,
		Shuffle:      true,
		UseGPU:       true,
		Normalise:    true,
		Distort:      true,
		WeightInit:   nnet.GlorotUniform,
	}
	// 1
	conf = conf.AddLayers(block1(64, 3)...)
	// 2
	conf = conf.AddLayers(block1(128, 3)...)
	conf = conf.AddLayers(block1(128, 3)...)
	conf = conf.AddLayers(block2(128, 3)...)
	// 5
	conf = conf.AddLayers(block1(128, 3)...)
	conf = conf.AddLayers(block1(128, 3)...)
	conf = conf.AddLayers(block2(256, 3)...)
	// 8
	conf = conf.AddLayers(block1(256, 3)...)
	conf = conf.AddLayers(block2(256, 3)...)
	// 10
	conf = conf.AddLayers(block1(512, 3)...)

	conf = conf.AddLayers(
		nnet.Conv{Nfeats: 2048, Size: 1, Pad: true},
		nnet.Activation{Atype: "relu"},
		nnet.Dropout{Ratio: 0.2},

		nnet.Conv{Nfeats: 256, Size: 1, Pad: true},
		nnet.Activation{Atype: "relu"},
		nnet.Pool{Size: 2},
		nnet.Dropout{Ratio: 0.2},

		nnet.Conv{Nfeats: 256, Size: 3, Pad: true},
		nnet.Activation{Atype: "relu"},
		nnet.Pool{Size: 2},

		nnet.Flatten{},
		nnet.Linear{Nout: 10},
		nnet.Activation{Atype: "softmax"},
	)
	fmt.Println(conf)
	err := conf.Save("cifar10_sim.conf")
	nnet.CheckErr(err)
}
