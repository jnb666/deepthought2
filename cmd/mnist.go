package main

import (
	"encoding/binary"
	"fmt"
	"github.com/jnb666/deepthought2/nnet"
	"os"
	"path"
)

const (
	classes  = 10
	maxPixel = 255
)

type label struct{ Magic, Num uint32 }

type image struct{ Magic, Num, Width, Height uint32 }

func main() {
	train, err := loadData("train-images-idx3-ubyte", "train-labels-idx1-ubyte")
	nnet.CheckErr(err)
	err = train.Save("mnist_train")
	nnet.CheckErr(err)

	test, err := loadData("t10k-images-idx3-ubyte", "t10k-labels-idx1-ubyte")
	nnet.CheckErr(err)
	err = test.Save("mnist_test")
	nnet.CheckErr(err)

	conf := nnet.Config{
		DataSet:       "mnist",
		Eta:           0.1,
		Lambda:        3.0,
		NormalWeights: true,
		FlattenInput:  true,
		MaxEpoch:      20,
		TrainBatch:    10,
		TestBatch:     100,
		Threads:       4,
		Shuffle:       true,
		Layers: []nnet.LayerConfig{
			nnet.Linear(100),
			nnet.Activation("relu"),
			nnet.Linear(10),
			nnet.LogRegression(),
		},
	}
	fmt.Println(conf)
	save("mnist", conf)

	conf.Layers = []nnet.LayerConfig{
		nnet.LinearDNN(100),
		nnet.ReluDNN(),
		nnet.LinearDNN(10),
		nnet.LogRegression(),
	}
	save("mnist_dnn", conf)
}

func save(model string, conf nnet.Config) {
	err := conf.SaveDefault(model)
	nnet.CheckErr(err)
	if !nnet.FileExists(model + ".net") {
		err = conf.Save(model)
		nnet.CheckErr(err)
	}
}

func loadData(imageFile, labelFile string) (d nnet.Data, err error) {
	d.Classes = classes
	d.Labels, err = readLabels(labelFile)
	if err != nil {
		return
	}
	d.Input, d.Shape, err = readImages(imageFile)
	return
}

func readImages(name string) ([]float32, []int, error) {
	pathName := path.Join(nnet.DataDir, "mnist", name)
	f, err := os.Open(pathName)
	if err != nil {
		return nil, nil, err
	}
	defer f.Close()
	var head image
	if err = binary.Read(f, binary.BigEndian, &head); err != nil {
		return nil, nil, err
	}
	n, h, w := int(head.Num), int(head.Height), int(head.Width)
	fmt.Printf("read %d %dx%d images from %s\n", n, h, w, name)
	images := make([]float32, n*h*w)
	batch := make([]byte, h*w)
	for i := 0; i < n; i++ {
		_, err = f.Read(batch)
		if err != nil {
			return nil, nil, err
		}
		for j, val := range batch {
			images[i*w*h+j] = float32(val) / maxPixel
		}
	}
	return images, []int{h, w}, err
}

func readLabels(name string) ([]int32, error) {
	pathName := path.Join(nnet.DataDir, "mnist", name)
	f, err := os.Open(pathName)
	if err != nil {
		return nil, err
	}
	defer f.Close()
	var head label
	if err = binary.Read(f, binary.BigEndian, &head); err != nil {
		return nil, err
	}
	fmt.Printf("read %d labels from %s\n", head.Num, name)
	bytes := make([]byte, head.Num)
	if _, err = f.Read(bytes); err != nil {
		return nil, err
	}
	labels := make([]int32, head.Num)
	for i, label := range bytes {
		labels[i] = int32(label)
	}
	return labels, err
}
