package main

import (
	"flag"
	"fmt"
	"github.com/jnb666/deepthought2/nnet"
	"github.com/jnb666/deepthought2/num"
	"log"
	"os"
)

func predict(q num.Queue, net *nnet.Network, d nnet.Data) {
	rng := nnet.SetSeed(net.RandSeed)
	dset := nnet.NewDataset(q.Dev(), d, net.DatasetConfig(false), rng)
	dset.NextEpoch()
	x, y, _ := dset.NextBatch()
	classes := q.NewArray(num.Int32, y.Dims[0])
	yPred := net.Fprop(x, false)
	q.Call(num.Unhot(yPred, classes))
	log.Print("predict:\n", yPred.String(q))
	log.Println("classes:", classes.String(q))
	log.Println("labels: ", y.String(q))
}

func main() {
	if len(os.Args) < 2 {
		fmt.Fprintln(os.Stderr, "Usage: train [opts] <model>")
		os.Exit(1)
	}
	model := os.Args[len(os.Args)-1]
	err := nnet.InitLogger(model, 0)
	nnet.CheckErr(err)
	log.Println("load model:", model)
	conf, err := nnet.LoadConfig(model + ".conf")
	nnet.CheckErr(err)

	// override config settings from command line
	flag.Float64Var(&conf.Eta, "eta", conf.Eta, "learning rate")
	flag.Float64Var(&conf.Lambda, "lambda", conf.Lambda, "weight decay parameter")
	flag.Float64Var(&conf.Momentum, "momentum", conf.Momentum, "momentum")
	flag.BoolVar(&conf.Nesterov, "nesterov", conf.Nesterov, "nesterov momentum")
	flag.Int64Var(&conf.RandSeed, "seed", conf.RandSeed, "random number seed")
	flag.IntVar(&conf.MaxEpoch, "epochs", conf.MaxEpoch, "max epochs")
	flag.IntVar(&conf.MaxSamples, "samples", conf.MaxSamples, "max samples")
	flag.IntVar(&conf.TrainBatch, "batch", conf.TrainBatch, "train batch size")
	flag.IntVar(&conf.TestBatch, "testbatch", conf.TestBatch, "test batch size")
	flag.IntVar(&conf.DebugLevel, "debug", conf.DebugLevel, "debug logging level")
	flag.BoolVar(&conf.Profile, "profile", conf.Profile, "print profiling info")
	flag.BoolVar(&conf.MemProfile, "memprofile", conf.Profile, "print memory profiling info")
	flag.BoolVar(&conf.UseGPU, "gpu", conf.UseGPU, "use Cuda GPU acceleration")
	flag.BoolVar(&conf.Normalise, "norm", conf.Normalise, "normalise input data")
	flag.BoolVar(&conf.Distort, "distort", conf.Distort, "apply image distortion")
	flag.Parse()

	dev := num.NewDevice(conf.UseGPU)
	q := dev.NewQueue()
	q.Profiling(conf.Profile, "main")
	rng := nnet.SetSeed(conf.RandSeed)

	// load traing and test data
	data, err := nnet.LoadData(conf.DataSet)
	nnet.CheckErr(err)
	trainData := nnet.NewDataset(dev, data["train"], conf.DatasetConfig(false), rng)
	trainData.Profiling(conf.Profile, "train")

	// create network and initialise weights
	trainNet := nnet.New(q, conf, trainData.BatchSize, trainData.Shape(), true, rng)
	log.Println(trainNet)
	trainNet.InitWeights(rng)
	if conf.DebugLevel >= 1 {
		log.Println("== Before ==")
		predict(q, trainNet, data["train"])
	}

	// train the network
	rng2 := nnet.SetSeed(conf.RandSeed)
	tester := nnet.NewTestLogger(dev, conf, data, rng2)
	nnet.Train(trainNet, trainData, tester)

	// exit
	if conf.DebugLevel >= 1 {
		log.Println("== After ==")
		predict(q, trainNet, data["train"])
	}
	if conf.MemProfile {
		log.Printf(trainNet.MemoryProfile())
		log.Printf(tester.MemoryProfile())
	} else {
		log.Printf("memory used: %s\n", nnet.FormatBytes(trainNet.Memory()+tester.Memory()))
	}
	if conf.Profile {
		log.Print(q.Profile())
	}
	trainNet.Release()
	trainData.Release()
	tester.Release()
	q.Shutdown()
}
