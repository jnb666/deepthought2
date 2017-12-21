package web

import (
	"github.com/gorilla/mux"
	"github.com/jnb666/deepthought2/img"
	"image/png"
	"math/rand"
	"net/http"
	"strconv"
)

type ImagePage struct {
	*Templates
	Dset    string
	Page    int
	Errors  bool
	Distort string
	Rows    []int
	Cols    []int
	Width   int
	Height  int
	Pages   int
	Total   int
	net     *Network
}

// Base data for handler functions to view input image dataset
func NewImagePage(t *Templates, net *Network, scale float64, rows, cols int) *ImagePage {
	p := &ImagePage{net: net, Templates: t, Page: 1}
	p.AddOption(Link{Name: "all", Url: "./all", Selected: true})
	p.AddOption(Link{Name: "errors", Url: "./errors"})
	p.AddOption(Link{Name: "prev", Url: "./prev"})
	p.AddOption(Link{Name: "next", Url: "./next"})
	p.AddOption(Link{Name: "distort", Url: "./distort"})
	dims := net.Data["train"].Shape()
	if len(dims) >= 2 {
		p.Width = int(float64(dims[1]) * scale)
		p.Height = int(float64(dims[0]) * scale)
		p.Rows = seq(rows)
		p.Cols = seq(cols)
	}
	return p
}

// Handler function for the main image page
func (p *ImagePage) Base() func(w http.ResponseWriter, r *http.Request) {
	return func(w http.ResponseWriter, r *http.Request) {
		p.net.Lock()
		defer p.net.Unlock()
		vars := mux.Vars(r)
		p.Dset = vars["dset"]
		p.Select("/images/" + p.Dset + "/")
		sel := []string{"all"}
		if p.Errors {
			sel = []string{"errors"}
		}
		if p.Distort != "" {
			sel = append(sel, "distort")
		}
		p.SelectOptions(sel)
		p.Heading = p.net.heading()
		template := "images"
		if _, ok := p.net.Data[p.Dset]; !ok {
			template = "blank"
		}
		p.Exec(w, template, p, true)
	}
}

// Handler function for the frame with grid of images
func (p *ImagePage) Grid() func(w http.ResponseWriter, r *http.Request) {
	return func(w http.ResponseWriter, r *http.Request) {
		p.net.Lock()
		defer p.net.Unlock()
		vars := mux.Vars(r)
		p.Dset = vars["dset"]
		p.Total, p.Pages = p.pageCount()
		if p.Page > p.Pages || p.Page < 1 {
			p.Page = 1
		}
		p.Exec(w, "grid", p, false)
	}
}

// Set option from top menu
func (p *ImagePage) Setopt() func(w http.ResponseWriter, r *http.Request) {
	return func(w http.ResponseWriter, r *http.Request) {
		p.net.Lock()
		defer p.net.Unlock()
		vars := mux.Vars(r)
		p.Dset = vars["dset"]
		p.Total, p.Pages = p.pageCount()
		switch vars["opt"] {
		case "all":
			p.Errors = false
		case "errors":
			p.Errors = true
		case "prev":
			p.Page = mod(p.Page-1, 1, p.Pages)
		case "next":
			p.Page = mod(p.Page+1, 1, p.Pages)
		case "distort":
			if p.Distort == "" {
				p.Distort = strconv.Itoa(rand.Intn(999999))
			} else {
				p.Distort = ""
			}
		}
		http.Redirect(w, r, "/images/"+p.Dset+"/", http.StatusFound)
	}
}

func (p *ImagePage) pageCount() (nimg, pages int) {
	if _, ok := p.net.Data[p.Dset]; !ok {
		return 0, 1
	}
	if p.Errors {
		nimg = p.errorImageCount(-1)
	} else {
		nimg = p.net.Data[p.Dset].Len()
	}
	rows, cols := len(p.Rows), len(p.Cols)
	pages = nimg / (rows * cols)
	if nimg%(rows*cols) != 0 {
		pages++
	}
	return nimg, pages
}

func (p *ImagePage) errorImageCount(index int) int {
	pred, ok := p.net.Pred[p.Dset]
	count := 0
	if ok {
		for i := range pred {
			if pred[i] != p.net.Labels[p.Dset][i] {
				count++
			}
			if index >= 0 && count == index+1 {
				return i + 1
			}
		}
	}
	if index < 0 {
		return count
	}
	return 0
}

func (p *ImagePage) Index(row, col int) int {
	rows, cols := len(p.Rows), len(p.Cols)
	index := (p.Page-1)*rows*cols + row*cols + col
	if index >= p.Total {
		return 0
	}
	if p.Errors {
		return p.errorImageCount(index)
	}
	return index + 1
}

func (p *ImagePage) Label(i int) int {
	lab := p.net.Labels[p.Dset]
	if i < 1 || i > len(lab) {
		return -1
	}
	return int(lab[i-1])
}

func (p *ImagePage) Predict(i int) int {
	pred, ok := p.net.Pred[p.Dset]
	if !ok || i < 1 || i > len(pred) {
		return -1
	}
	return int(pred[i-1])
}

// Handler function for the image data
func (p *ImagePage) Image() func(w http.ResponseWriter, r *http.Request) {
	return func(w http.ResponseWriter, r *http.Request) {
		p.net.Lock()
		defer p.net.Unlock()
		vars := mux.Vars(r)
		dset := vars["dset"]
		id, _ := strconv.Atoi(vars["id"])
		data, ok := p.net.Data[dset]
		if !ok || id < 1 || id > data.Len() {
			http.NotFound(w, r)
			return
		}
		image := data.Image(id - 1)
		if r.FormValue("d") != "" {
			image = p.net.trans.Transform(image, 0)
		}
		pred := p.Predict(id)
		image = img.Highlight(image, pred >= 0 && p.Label(id) != pred)
		w.Header().Set("Content-type", "image/png")
		png.Encode(w, image)
	}
}

func seq(n int) []int {
	s := make([]int, n)
	for i := range s {
		s[i] = i
	}
	return s
}

func mod(i, min, max int) int {
	if i < min {
		i = max
	}
	if i > max {
		i = min
	}
	return i
}
