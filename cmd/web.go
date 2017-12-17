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
	net, err := web.NewNetwork(model)
	nnet.CheckErr(err)

	t, err := web.NewTemplates()
	nnet.CheckErr(err)
	t.AddMenuItem(web.Link{Url: "/train", Name: "train net"})
	for _, key := range nnet.DataTypes {
		t.AddMenuItem(web.Link{Url: "/images/" + key + "/", Name: key + " images"})
	}
	t.AddMenuItem(web.Link{Url: "/config", Name: "edit config"})

	r := mux.NewRouter()
	r.Handle("/", http.RedirectHandler("/train/stats", http.StatusFound))
	r.PathPrefix("/static/").Handler(http.FileServer(http.Dir(web.AssetDir)))

	trainPage := web.NewTrainPage(t.Clone(), net)
	r.HandleFunc("/train", trainPage.Base())
	r.HandleFunc("/train/{cmd}", trainPage.Command())
	r.HandleFunc("/stats", trainPage.Stats())
	r.HandleFunc("/ws", trainPage.Websocket())

	imagePage := web.NewImagePage(t.Clone(), net, scale, rows, cols)
	r.HandleFunc("/images/{dset}/", imagePage.Base())
	r.HandleFunc("/images/{dset}/{opt}", imagePage.Setopt())
	r.HandleFunc("/grid/{dset}", imagePage.Grid())
	r.HandleFunc("/img/{dset}/{id:[0-9]+}", imagePage.Image())

	configPage := web.NewConfigPage(t.Clone(), net)
	r.HandleFunc("/config", configPage.Base())
	r.HandleFunc("/config/load", configPage.Load())
	r.HandleFunc("/config/save", configPage.Save()).Methods("POST")
	r.HandleFunc("/config/reset", configPage.Reset())

	r.Handle("/favicon.ico", http.NotFoundHandler())
	r.NotFoundHandler = t.ErrorHandler(http.StatusNotFound, web.ErrNotFound)

	fmt.Println("serving web page at http://localhost:8080")
	http.ListenAndServe(":8080", r)
}
