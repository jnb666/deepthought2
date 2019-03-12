// Residual network: see https://arxiv.org/abs/1512.03385
package main

import (
	"fmt"
	"github.com/jnb666/deepthought2/nnet"
)

type opts struct {
	name    string
	stack   int
	width   int
	dropout float64
}

// depth = 6*stack + 4
var options = []opts{
	{name: "res20", stack: 3, width: 1},
	{name: "res32", stack: 5, width: 1},
	{name: "wide16_8", stack: 2, width: 8},
	{name: "wide16_8d", stack: 2, width: 8, dropout: 0.3},
	{name: "wide28_10", stack: 4, width: 10},
	{name: "wide28_10d", stack: 4, width: 10, dropout: 0.3},
}

func block(nfeat, stride int, dropout float64, prefix bool) []nnet.ConfigLayer {
	var blk []nnet.ConfigLayer
	if prefix {
		blk = append(blk,
			nnet.BatchNorm{},
			nnet.Activation{Atype: "relu"},
		)
	}
	blk = append(blk,
		nnet.Conv{Nfeats: nfeat, Size: 3, Stride: stride, Pad: true, NoBias: true},
		nnet.BatchNorm{},
		nnet.Activation{Atype: "relu"},
	)
	if dropout > 0 {
		blk = append(blk, nnet.Dropout{Ratio: dropout})
	}
	return append(blk, nnet.Conv{Nfeats: nfeat, Size: 3, Pad: true, NoBias: true})
}

func addBlock(c nnet.Config, nin, nout, stride int, dropout float64) nnet.Config {
	if stride == 1 && nin == nout {
		c = c.AddLayers(
			nnet.AddLayer(block(nout, 1, dropout, true), nil),
		)
	} else {
		project := []nnet.ConfigLayer{
			nnet.Conv{Nfeats: nout, Size: 1, Stride: stride, Pad: true, NoBias: true},
		}
		c = c.AddLayers(
			nnet.BatchNorm{},
			nnet.Activation{Atype: "relu"},
			nnet.AddLayer(block(nout, stride, dropout, false), project),
		)
	}
	return c
}

func main() {
	conf := nnet.Config{
		DataSet:      "cifar10_2",
		Eta:          0.05,
		EtaDecay:     0.2,
		EtaDecayStep: 25,
		Lambda:       250,
		Momentum:     0.9,
		Nesterov:     true,
		MaxEpoch:     200,
		ValidEMA:     20,
		StopAfter:    3,
		ExtraEpochs:  4,
		TrainBatch:   125,
		Shuffle:      true,
		UseGPU:       true,
		Normalise:    true,
		Distort:      true,
		FastConv:     true,
		WeightInit:   nnet.HeNormal,
	}

	for _, opt := range options {
		c := conf.AddLayers(nnet.Conv{Nfeats: 16, Size: 3, Pad: true, NoBias: true})
		// [32,32,16] => [32,32,16k]
		k := 16 * opt.width
		c = addBlock(c, 16, k, 1, opt.dropout)
		for i := 1; i < opt.stack; i++ {
			c = addBlock(c, k, k, 1, opt.dropout)
		}
		// [32,32,16k] => [16,16,32k]
		c = addBlock(c, k, 2*k, 2, opt.dropout)
		for i := 1; i < opt.stack; i++ {
			c = addBlock(c, 2*k, 2*k, 1, opt.dropout)
		}
		// [16,16,32k] => [8,8,64k]
		c = addBlock(c, 2*k, 4*k, 2, opt.dropout)
		for i := 1; i < opt.stack; i++ {
			c = addBlock(c, 4*k, 4*k, 1, opt.dropout)
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
