package main

import (
	"encoding/binary"
	"fmt"
	"image/color"
	"os"
	"path"

	"github.com/jnb666/deepthought2/img"
	"github.com/jnb666/deepthought2/nnet"
)

const (
	epochs = 25
	split  = 50000
)

var classes = []string{"0", "1", "2", "3", "4", "5", "6", "7", "8", "9"}

type labelHeader struct{ Magic, Num uint32 }

type imageHeader struct{ Magic, Num, Width, Height uint32 }

func main() {
	// mnist dataset is 60000 train + 10000 test images
	train, err := loadData("train-images-idx3-ubyte", "train-labels-idx1-ubyte")
	nnet.CheckErr(err)
	test, err := loadData("t10k-images-idx3-ubyte", "t10k-labels-idx1-ubyte")
	nnet.CheckErr(err)

	mean, std := img.GetStats(train.Images, test.Images)
	train.Mean, train.StdDev = mean, std
	test.Mean, test.StdDev = mean, std

	err = nnet.SaveDataFile(train, "mnist_train")
	nnet.CheckErr(err)
	err = nnet.SaveDataFile(test, "mnist_test")
	nnet.CheckErr(err)

	// mnist2 dataset is 50000 train + 10000 valid + 10000 test images
	err = nnet.SaveDataFile(train.Slice(0, split), "mnist2_train")
	nnet.CheckErr(err)

	err = nnet.SaveDataFile(train.Slice(split, train.Len()), "mnist2_valid")
	nnet.CheckErr(err)

	err = nnet.SaveDataFile(test, "mnist2_test")
	nnet.CheckErr(err)
}

func loadData(imageFile, labelFile string) (*img.Data, error) {
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

func readImages(name string) (images []*img.Image, err error) {
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
	images = make([]*img.Image, n)
	pixels := make([]uint8, w*h)
	for i := range images {
		if _, err = f.Read(pixels); err != nil {
			return
		}
		img := img.NewImage(w, h, 1)
		for j, pix := range pixels {
			img.Set(j%w, j/w, color.Gray{Y: pix})
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
