package main

import (
	"flag"
	"fmt"
	"github.com/jnb666/deepthought2/nnet"
	"github.com/jnb666/deepthought2/num"
	"os"
)

func predict(q num.Queue, net *nnet.Network, dset *nnet.Dataset) {
	x, y, _ := dset.GetBatch(q, 0)
	classes := q.NewArray(num.Int32, y.Dims()[0])
	yPred := net.Predict(x, classes)
	fmt.Print("predict:", yPred.String(q))
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
	conf, err := nnet.LoadConfig(model + ".net")
	nnet.CheckErr(err)

	// override config settings from command line
	flag.Float64Var(&conf.Eta, "eta", conf.Eta, "learning rate")
	flag.Float64Var(&conf.Lambda, "lambda", conf.Lambda, "weight decay parameter")
	flag.Int64Var(&conf.RandSeed, "seed", conf.RandSeed, "random number seed")
	flag.IntVar(&conf.MaxEpoch, "epochs", conf.MaxEpoch, "max epochs")
	flag.IntVar(&conf.MaxSamples, "samples", conf.MaxSamples, "max samples")
	flag.IntVar(&conf.TrainBatch, "batch", conf.TrainBatch, "train batch size")
	flag.IntVar(&conf.TestBatch, "testbatch", conf.TestBatch, "test batch size")
	flag.IntVar(&conf.DebugLevel, "debug", conf.DebugLevel, "debug logging level")
	flag.BoolVar(&conf.Profile, "profile", conf.Profile, "print profiling info")
	flag.Parse()

	dev := num.NewCPUDevice()
	q := dev.NewQueue(conf.Threads)

	// load traing and test data
	data, err := nnet.LoadData(conf.DataSet)
	nnet.CheckErr(err)
	trainData := nnet.NewDataset(dev, data["train"], conf.TrainBatch, conf.MaxSamples)

	// initialise weights
	q.Profiling(conf.Profile)
	trainNet := nnet.New(q, conf, trainData.BatchSize, trainData.Shape)
	fmt.Println(trainNet)
	trainNet.InitWeights()
	if conf.DebugLevel >= 1 {
		fmt.Println("== Before ==")
		predict(q, trainNet, trainData)
	}

	// train the network
	tester := nnet.NewTestLogger(q, conf, data)
	nnet.Train(trainNet, trainData, tester)

	if conf.DebugLevel >= 1 {
		fmt.Println("== After ==")
		predict(q, trainNet, trainData)
	}
	q.Shutdown()
}
