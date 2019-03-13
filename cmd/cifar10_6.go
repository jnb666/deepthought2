// Simple 6 layer network based on https://www.kaggle.com/c/cifar-10/discussion/40237
package main

import (
	"fmt"

	"github.com/jnb666/deepthought2/nnet"
)

func block(nfeat int, dropout float64) []nnet.ConfigLayer {
	return []nnet.ConfigLayer{
		nnet.Conv{Nfeats: nfeat, Size: 3, Pad: true},
		nnet.Activation{Atype: "relu"},
		nnet.Conv{Nfeats: nfeat, Size: 3, Pad: true},
		nnet.Activation{Atype: "relu"},
		nnet.Pool{Size: 2},
		nnet.Dropout{Ratio: dropout},
	}
}

func blockBN(nfeat int, dropout float64) []nnet.ConfigLayer {
	return []nnet.ConfigLayer{
		nnet.Conv{Nfeats: nfeat, Size: 3, Pad: true, NoBias: true},
		nnet.BatchNorm{},
		nnet.Activation{Atype: "relu"},
		nnet.Conv{Nfeats: nfeat, Size: 3, Pad: true, NoBias: true},
		nnet.BatchNorm{},
		nnet.Activation{Atype: "relu"},
		nnet.Pool{Size: 2},
		nnet.Dropout{Ratio: dropout},
	}
}

func output() []nnet.ConfigLayer {
	return []nnet.ConfigLayer{
		nnet.Flatten{},
		nnet.Linear{Nout: 10},
		nnet.Activation{Atype: "softmax"},
	}
}

func main() {
	conf := nnet.Config{
		DataSet:     "cifar10_2",
		Adam:        true,
		Eta:         0.001,
		Lambda:      10,
		MaxEpoch:    200,
		StopAfter:   2,
		ExtraEpochs: 4,
		ValidEMA:    20,
		TrainBatch:  125,
		Shuffle:     true,
		UseGPU:      true,
		Normalise:   true,
		Distort:     true,
		WeightInit:  nnet.GlorotUniform,
	}
	c := conf.AddLayers(block(32, 0.2)...)
	c = c.AddLayers(block(64, 0.3)...)
	c = c.AddLayers(block(128, 0.4)...)
	c = c.AddLayers(output()...)
	fmt.Println(c)
	err := c.Save("cifar10_6.conf")
	nnet.CheckErr(err)

	c = conf.AddLayers(blockBN(32, 0.2)...)
	c = c.AddLayers(blockBN(64, 0.3)...)
	c = c.AddLayers(blockBN(128, 0.4)...)
	c = c.AddLayers(output()...)
	fmt.Println(c)
	err = c.Save("cifar10_6n.conf")
	nnet.CheckErr(err)
}
