package main

import (
	"encoding/binary"
	"fmt"
	"github.com/jnb666/deepthought2/img"
	"github.com/jnb666/deepthought2/nnet"
	"image"
	"os"
	"path"
)

const classes = 10

type labelHeader struct{ Magic, Num uint32 }

type imageHeader struct{ Magic, Num, Width, Height uint32 }

func main() {
	train, err := loadData("train-images-idx3-ubyte", "train-labels-idx1-ubyte")
	nnet.CheckErr(err)
	err = nnet.SaveDataFile(train, "mnist_train")
	nnet.CheckErr(err)

	test, err := loadData("t10k-images-idx3-ubyte", "t10k-labels-idx1-ubyte")
	nnet.CheckErr(err)
	err = nnet.SaveDataFile(test, "mnist_test")
	nnet.CheckErr(err)
}

func loadData(imageFile, labelFile string) (nnet.Data, error) {
	labels, err := readLabels(labelFile)
	if err != nil {
		return nil, err
	}
	images, err := readImages(imageFile)
	if err != nil {
		return nil, err
	}
	return img.NewData(classes, labels, images), nil
}

func readImages(name string) (images []image.Image, err error) {
	var f *os.File
	pathName := path.Join(nnet.DataDir, "mnist", name)
	if f, err = os.Open(pathName); err != nil {
		return
	}
	defer f.Close()
	var head imageHeader
	if err = binary.Read(f, binary.BigEndian, &head); err != nil {
		return
	}
	n, h, w := int(head.Num), int(head.Height), int(head.Width)
	fmt.Printf("read %d %dx%d images from %s\n", n, h, w, name)
	images = make([]image.Image, n)
	shape := image.Rect(0, 0, w, h)
	for i := range images {
		img := image.NewGray(shape)
		if _, err = f.Read(img.Pix); err != nil {
			return
		}
		images[i] = img
	}
	return
}

func readLabels(name string) (labels []int32, err error) {
	var f *os.File
	pathName := path.Join(nnet.DataDir, "mnist", name)
	if f, err = os.Open(pathName); err != nil {
		return
	}
	defer f.Close()
	var head labelHeader
	if err = binary.Read(f, binary.BigEndian, &head); err != nil {
		return
	}
	fmt.Printf("read %d labels from %s\n", head.Num, name)
	bytes := make([]byte, head.Num)
	if _, err = f.Read(bytes); err != nil {
		return
	}
	labels = make([]int32, head.Num)
	for i, label := range bytes {
		labels[i] = int32(label)
	}
	return
}
