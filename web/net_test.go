package web

import (
	"github.com/jnb666/deepthought2/nnet"
	"testing"
)

func TestRunConfig(t *testing.T) {
	conf, err := nnet.LoadConfig("mnist_mlp.conf")
	if err != nil {
		t.Error(err)
	}
	param := []TuneParams{
		{Name: "Eta", Values: []string{"0.1", "0.05", "0.15"}},
		{Name: "Lambda", Values: []string{"3", "5"}},
		{Name: "TrainBatch", Values: []string{"10", "20"}},
	}
	runs := getRunConfig(conf, param)
	if len(runs) != 12 {
		t.Errorf("got %d runs expect 12", len(runs))
	}
}
