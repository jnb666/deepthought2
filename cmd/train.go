package main

import (
	"flag"
	"fmt"
	"github.com/jnb666/deepthought2/nnet"
	"github.com/jnb666/deepthought2/num"
	"os"
)

func predict(q num.Queue, net *nnet.Network, d nnet.Data) {
	dset := nnet.NewDataset(q.Dev(), d, 10, 10, net.FlattenInput, nil)
	dset.Rewind()
	x, y, _ := dset.NextBatch()
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
	conf, err := nnet.LoadConfig(model + ".conf")
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
	flag.BoolVar(&conf.UseGPU, "gpu", conf.UseGPU, "use Cuda GPU acceleration")
	flag.Parse()

	dev := num.NewDevice(conf.UseGPU)
	q := dev.NewQueue()
	q.Profiling(conf.Profile)
	rng := nnet.SetSeed(conf.RandSeed)

	// load traing and test data
	data, err := nnet.LoadData(conf.DataSet)
	nnet.CheckErr(err)
	trainData := nnet.NewDataset(dev, data["train"], conf.TrainBatch, conf.MaxSamples, conf.FlattenInput, rng)

	// create network and initialise weights
	trainNet := nnet.New(q, conf, trainData.BatchSize, trainData.Shape(), rng)
	fmt.Println(trainNet)
	trainNet.InitWeights(rng)
	if conf.DebugLevel >= 1 {
		fmt.Println("== Before ==")
		predict(q, trainNet, data["train"])
	}

	// train the network
	rng2 := nnet.SetSeed(conf.RandSeed)
	tester := nnet.NewTestLogger(q, conf, data, rng2)
	nnet.Train(trainNet, trainData, tester)

	// exit
	if conf.DebugLevel >= 1 {
		fmt.Println("== After ==")
		predict(q, trainNet, data["train"])
	}
	q.Shutdown()
	if conf.Profile {
		fmt.Printf("== Profile ==\n%s\n", q.Profile())
	}
}
