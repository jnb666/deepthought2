package main

import (
	"fmt"
	"github.com/gorilla/mux"
	"github.com/jnb666/deepthought2/nnet"
	"github.com/jnb666/deepthought2/web"
	"log"
	"net/http"
	"os"
)

const (
	rows   = 7
	cols   = 10
	width  = 1200
	height = 900
)

func main() {
	log.SetFlags(0)
	if len(os.Args) < 2 {
		fmt.Println("usage: web [opts] <model>")
		os.Exit(1)
	}
	model := os.Args[len(os.Args)-1]
	net, err := web.NewNetwork(model)
	nnet.CheckErr(err)

	t, err := web.NewTemplates()
	nnet.CheckErr(err)

	t.AddMenuItem("", "view")
	t.AddMenuItem("/train", "stats")
	for _, key := range nnet.DataTypes {
		t.AddMenuItem("/images/"+key+"/", key+" images")
	}
	t.AddMenuItem("/view/outputs/", "activations")
	t.AddMenuItem("/view/weights/", "weights")
	t.AddMenuItem("/config", "settings")

	t.AddMenuItem("", "train")
	for _, opt := range []string{"start", "stop", "continue", "reset"} {
		t.AddMenuItem("/train/"+opt, opt)
	}

	r := mux.NewRouter()
	r.Handle("/", http.RedirectHandler("/train/stats", http.StatusFound))
	r.PathPrefix("/static/").Handler(http.FileServer(http.Dir(web.AssetDir)))

	trainPage := web.NewTrainPage(t.Clone(), net)
	r.HandleFunc("/train", trainPage.Base())
	r.HandleFunc("/train/{cmd:[a-z]+}", trainPage.Command())
	r.HandleFunc("/train/set/{opt:[a-z]+}", trainPage.Setopt())
	r.HandleFunc("/stats", trainPage.Stats())
	r.HandleFunc("/stats/update", trainPage.Filter()).Methods("POST")
	r.HandleFunc("/ws", trainPage.Websocket())

	imagePage := web.NewImagePage(t.Clone(), net, rows, cols, height, width)
	r.HandleFunc("/images/{dset}/{class:[0-9]*}", imagePage.Base())
	r.HandleFunc("/images/{dset}/{opt:[a-z]+}", imagePage.Setopt())
	r.HandleFunc("/grid/{dset}", imagePage.Grid())
	r.HandleFunc("/img/{dset}/{id:[0-9]+}/", imagePage.Image())
	r.HandleFunc("/img/{dset}/{id:[0-9]+}/{opts}", imagePage.Image())

	viewPage := web.NewViewPage(t.Clone(), net)
	r.HandleFunc("/view/{page}/", viewPage.Base())
	r.HandleFunc("/view/{page}/{opt:[a-z]+}", viewPage.Setopt())
	r.HandleFunc("/net/{page}", viewPage.Network())
	r.HandleFunc("/net/{page}/{layer:[0-9]+}", viewPage.Image())

	configPage := web.NewConfigPage(t.Clone(), net)
	r.HandleFunc("/config", configPage.Base())
	r.HandleFunc("/config/load", configPage.Load())
	r.HandleFunc("/config/save", configPage.Save()).Methods("POST")
	r.HandleFunc("/config/reset", configPage.Reset())
	r.HandleFunc("/config/tune", configPage.Tune()).Methods("POST")

	r.Handle("/favicon.ico", http.NotFoundHandler())
	r.NotFoundHandler = t.ErrorHandler(http.StatusNotFound, web.ErrNotFound)

	fmt.Println("serving web page at http://localhost:8080")
	http.ListenAndServe(":8080", r)
}
