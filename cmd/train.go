package main

import (
	"flag"
	"fmt"
	"github.com/jnb666/deepthought2/nnet"
	"github.com/jnb666/deepthought2/num"
	"os"
)

func predict(q num.Queue, net *nnet.Network, data nnet.Data) {
	dset := nnet.NewDataset(q.Device(), data, net.TrainBatch, net.MaxSamples)
	x, y, _ := dset.GetBatch(q, 0)
	classes := num.NewArray(q.Device(), num.Int32, y.Dims()[0])
	yPred := net.Predict(q, x, classes)
	if net.DebugLevel == 0 && yPred.Dims()[1] == 1 {
		fmt.Println("predict:", yPred.Reshape(-1).String(q))
	}
	fmt.Println("classes:", classes.String(q))
	fmt.Println("labels: ", y.String(q))
}

func main() {
	if len(os.Args) < 2 {
		fmt.Println("Usage: train [opts] <model>")
		os.Exit(1)
	}
	model := os.Args[len(os.Args)-1]
	fmt.Println("load model:", model)
	conf, err := nnet.LoadConfig(model)
	nnet.CheckErr(err)

	// override config settings from command line
	flag.Float64Var(&conf.Eta, "eta", conf.Eta, "learning rate")
	flag.Float64Var(&conf.Lambda, "lambda", conf.Lambda, "weight decay parameter")
	flag.Int64Var(&conf.RandSeed, "seed", conf.RandSeed, "random number seed")
	flag.IntVar(&conf.MaxEpoch, "epochs", conf.MaxEpoch, "max epochs")
	flag.IntVar(&conf.MaxSamples, "samples", conf.MaxSamples, "max samples")
	flag.IntVar(&conf.DebugLevel, "debug", conf.DebugLevel, "debug logging level")
	flag.Parse()
	nnet.SetSeed(conf.RandSeed)

	// load traing and test data
	data, err := nnet.LoadData(model)
	nnet.CheckErr(err)

	// initialise weights
	net := nnet.New(conf)
	q := num.NewQueue(num.CPU, conf.Threads)
	net.InitWeights(q, data["train"].Shape)
	fmt.Println(net)

	fmt.Println("== Before ==")
	predict(q, net, data["train"])

	// train the network
	tester := nnet.NewTestLogger(num.CPU, data, conf)
	nnet.Train(q, net, data["train"], tester)

	fmt.Println("== After ==")
	predict(q, net, data["train"])
	q.Shutdown()
}
