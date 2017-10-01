package main

import (
	"flag"
	"fmt"
	"github.com/gorilla/mux"
	"github.com/jnb666/deepthought2/nnet"
	"github.com/jnb666/deepthought2/web"
	"log"
	"net/http"
	"os"
)

const (
	scale = 3
	rows  = 8
	cols  = 10
)

func main() {
	log.SetFlags(0)
	if len(os.Args) < 2 {
		fmt.Println("usage: web [opts] <model>")
		os.Exit(1)
	}
	model := os.Args[len(os.Args)-1]
	conf, err := web.NewConfig(model)
	nnet.CheckErr(err)
	flag.BoolVar(&conf.UseGPU, "gpu", conf.UseGPU, "use Cuda GPU acceleration")
	flag.Parse()

	net, err := web.NewNetwork(conf)
	nnet.CheckErr(err)

	t, err := web.NewTemplates()
	nnet.CheckErr(err)

	trainPage := web.NewTrainPage(t.Clone(), net)
	imagePage := web.NewImagePage(t.Clone(), net, scale, rows, cols)
	configPage := web.NewConfigPage(t.Clone(), conf)

	r := mux.NewRouter()
	r.Handle("/", http.RedirectHandler("/train/stats", http.StatusFound))
	r.PathPrefix("/static/").Handler(http.FileServer(http.Dir(web.AssetDir)))

	r.Handle("/train", http.RedirectHandler("/train/stats", http.StatusFound))
	r.HandleFunc("/train/{cmd:(?:stats|start|stop|continue)}", trainPage.Base())
	r.HandleFunc("/stats", trainPage.Stats())
	r.HandleFunc("/ws", trainPage.Websocket())

	r.Handle("/images", http.RedirectHandler("/images/all/train/1", http.StatusFound))
	r.HandleFunc("/images/{inc:(?:all|errors)}/{dset}/{page:[0-9]+}", imagePage.Base())
	r.HandleFunc("/img/{dset}/{id:[0-9]+}", imagePage.Image())

	r.HandleFunc("/config", configPage.Base())
	r.HandleFunc("/config/load", configPage.Load())
	r.HandleFunc("/config/save", configPage.Save()).Methods("POST")
	r.HandleFunc("/config/reset", configPage.Reset())

	fmt.Println("serving web page at http://localhost:8080")
	http.ListenAndServe(":8080", r)
}
