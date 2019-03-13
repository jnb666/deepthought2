package main

import (
	"flag"
	"fmt"
	"log"
	"os"

	"github.com/jnb666/deepthought2/nnet"
	"github.com/jnb666/deepthought2/num"
)

func predict(q num.Queue, net *nnet.Network, d nnet.Data) {
	rng := nnet.SetSeed(net.RandSeed)
	dset := nnet.NewDataset(q.Dev(), d, net.DatasetConfig(false), rng)
	dset.NextEpoch()
	x, y, _ := dset.NextBatch()
	classes := q.NewArray(num.Int32, y.Dims[0])
	yPred := nnet.Fprop(q, net.Layers, x, net.WorkSpace[0], false)
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
	flag.BoolVar(&conf.RMSprop, "rmsprop", conf.RMSprop, "use rmsprop optimiser")
	flag.BoolVar(&conf.Adam, "adam", conf.Adam, "use adam optimiser")
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
	defer q.Shutdown()
	q.Profiling(conf.Profile, "CPU")
	rng := nnet.SetSeed(conf.RandSeed)

	// load traing and test data
	data, err := nnet.LoadData(conf.DataSet)
	nnet.CheckErr(err)
	trainData := nnet.NewDataset(dev, data["train"], conf.DatasetConfig(false), rng)
	defer trainData.Release()

	// create network and initialise weights
	trainNet := nnet.New(q, conf, trainData.BatchSize, trainData.Shape(), true, rng)
	defer trainNet.Release()
	log.Println(trainNet)
	trainNet.InitWeights(rng)
	if conf.MaxEpoch < 1 {
		return
	}
	if conf.DebugLevel >= 1 {
		log.Println("== Before ==")
		predict(q, trainNet, data["train"])
	}

	// train the network
	rng2 := nnet.SetSeed(conf.RandSeed)
	tester := nnet.NewTestLogger(dev, conf, data, rng2)
	defer tester.Release()
	nnet.Train(trainNet, trainData, tester)

	if conf.DebugLevel >= 1 {
		log.Println("== After ==")
		predict(q, trainNet, data["train"])
	}

	nnet.MemoryProfile(conf.MemProfile, trainNet, tester.Network())
	if conf.Profile {
		log.Print(q.Profile())
	}
}
