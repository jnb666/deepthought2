// Residual network: see https://arxiv.org/abs/1512.03385
package main

import (
	"fmt"
	"github.com/jnb666/deepthought2/nnet"
)

type opts struct {
	name  string
	stack int
	width int
}

var options = []opts{
	{name: "res", stack: 3, width: 1},
	{name: "wide", stack: 2, width: 8},
}

func block(nfeat, stride int) []nnet.ConfigLayer {
	return []nnet.ConfigLayer{
		nnet.BatchNorm{},
		nnet.Activation{Atype: "relu"},
		nnet.Conv{Nfeats: nfeat, Size: 3, Stride: stride, Pad: true, NoBias: true},
		nnet.BatchNorm{},
		nnet.Activation{Atype: "relu"},
		nnet.Conv{Nfeats: nfeat, Size: 3, Pad: true, NoBias: true},
	}
}

func resBlock(nin, nout, stride int) nnet.ConfigLayer {
	if stride == 1 && nin == nout {
		return nnet.AddLayer(block(nout, 1), nil)
	}
	project := []nnet.ConfigLayer{
		nnet.Conv{Nfeats: nout, Size: 1, Stride: stride, Pad: true, NoBias: true},
	}
	return nnet.AddLayer(block(nout, stride), project)
}

func main() {
	conf := nnet.Config{
		DataSet:      "cifar10_2",
		Eta:          0.02,
		EtaDecay:     0.2,
		EtaDecayStep: 25,
		Lambda:       50,
		Momentum:     0.9,
		Nesterov:     true,
		MaxEpoch:     200,
		ValidEMA:     15,
		StopAfter:    2,
		ExtraEpochs:  4,
		TrainBatch:   125,
		Shuffle:      true,
		UseGPU:       true,
		Normalise:    true,
		Distort:      true,
		FastConv:     true,
		WeightInit:   nnet.GlorotUniform,
	}

	for _, opt := range options {
		c := conf.AddLayers(nnet.Conv{Nfeats: 16, Size: 3, Pad: true, NoBias: true})
		// [32,32,16] => [32,32,16k]
		k := 16 * opt.width
		c = c.AddLayers(resBlock(16, k, 1))
		for i := 1; i < opt.stack; i++ {
			c = c.AddLayers(resBlock(k, k, 1))
		}
		// [32,32,16k] => [16,16,32k]
		c = c.AddLayers(resBlock(k, 2*k, 2))
		for i := 1; i < opt.stack; i++ {
			c = c.AddLayers(resBlock(2*k, 2*k, 1))
		}
		// [16,16,32k] => [8,8,64k]
		c = c.AddLayers(resBlock(2*k, 4*k, 2))
		for i := 1; i < opt.stack; i++ {
			c = c.AddLayers(resBlock(4*k, 4*k, 1))
		}
		c = c.AddLayers(
			nnet.BatchNorm{},
			nnet.Activation{Atype: "relu"},
			nnet.Pool{Size: 8, Average: true},
			nnet.Flatten{},
			nnet.Linear{Nout: 10},
			nnet.Activation{Atype: "softmax"},
		)
		fmt.Println(c)
		err := c.Save("cifar10_" + opt.name + ".conf")
		nnet.CheckErr(err)

	}
}
