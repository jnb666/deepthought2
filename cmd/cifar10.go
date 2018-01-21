package main

import (
	"bufio"
	"fmt"
	"github.com/jnb666/deepthought2/img"
	"github.com/jnb666/deepthought2/nnet"
	"image/color"
	"io"
	"os"
	"path"
	"strings"
)

const (
	batchSize   = 10000
	imageWidth  = 32
	imageHeight = 32
	imageSize   = imageWidth * imageHeight
	imageBytes  = imageSize*3 + 1
	testSplit   = 5000
)

func main() {
	classes, err := readClasses("batches.meta.txt")
	nnet.CheckErr(err)

	train, err := loadBatch("data_batch_1.bin", classes)
	nnet.CheckErr(err)
	for i := 2; i <= 5; i++ {
		d, err := loadBatch(fmt.Sprintf("data_batch_%d.bin", i), classes)
		nnet.CheckErr(err)
		train.Labels = append(train.Labels, d.Labels...)
		train.Images = append(train.Images, d.Images...)
	}
	test, err := loadBatch("test_batch.bin", classes)
	nnet.CheckErr(err)

	mean, std := img.GetStats(train.Images, test.Images)
	train.Mean, train.StdDev = mean, std
	test.Mean, test.StdDev = mean, std

	err = nnet.SaveDataFile(train, "cifar10_train")
	nnet.CheckErr(err)

	err = nnet.SaveDataFile(test, "cifar10_test")
	nnet.CheckErr(err)

	err = nnet.SaveDataFile(train, "cifar10_2_train")
	nnet.CheckErr(err)

	err = nnet.SaveDataFile(test.Slice(0, testSplit), "cifar10_2_test")
	nnet.CheckErr(err)

	err = nnet.SaveDataFile(test.Slice(testSplit, test.Len()), "cifar10_2_valid")
	nnet.CheckErr(err)
}

// load batch of cifar-10 images and labels in bindary format
func loadBatch(name string, classes []string) (*img.Data, error) {
	pathName := path.Join(nnet.DataDir, "cifar-10", name)
	f, err := os.Open(pathName)
	if err != nil {
		return nil, err
	}
	defer f.Close()
	labels := make([]int32, batchSize)
	images := make([]*img.Image, batchSize)
	bytes := make([]uint8, imageBytes)
	for i := range labels {
		n, err := f.Read(bytes)
		if err == io.EOF {
			break
		}
		if err != nil {
			return nil, fmt.Errorf("error reading from %s: %s", pathName, err)
		}
		if n != imageBytes {
			return nil, fmt.Errorf("incomplete read: expected %d bytes got %d", imageBytes, n)
		}
		m := img.NewImage(imageWidth, imageHeight, 3)
		for j := 0; j < imageSize; j++ {
			col := color.NRGBA{R: bytes[1+j], G: bytes[1+imageSize+j], B: bytes[1+imageSize*2+j], A: 255}
			m.Set(j%imageWidth, j/imageWidth, col)
		}
		images[i] = m
		labels[i] = int32(bytes[0])
	}
	fmt.Printf("read %d images from %s\n", batchSize, name)
	d := img.NewData(classes, labels, images)
	return d, nil
}

// load class descriptions from file
func readClasses(name string) ([]string, error) {
	pathName := path.Join(nnet.DataDir, "cifar-10", name)
	f, err := os.Open(pathName)
	if err != nil {
		return nil, err
	}
	defer f.Close()
	s := bufio.NewScanner(f)
	classes := []string{}
	for s.Scan() {
		line := strings.TrimSpace(s.Text())
		if line != "" {
			classes = append(classes, s.Text())
		}
	}
	return classes, s.Err()
}
