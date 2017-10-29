package web

import (
	"fmt"
	"github.com/gorilla/mux"
	"github.com/jnb666/deepthought2/img"
	"github.com/jnb666/deepthought2/nnet"
	"image/png"
	"math/rand"
	"net/http"
	"strconv"
)

type ImagePage struct {
	*Templates
	Heading string
	Dset    string
	Rows    []int
	Cols    []int
	Width   int
	Height  int
	Distort string
	net     *Network
	errors  bool
	pages   int
	page    int
	nimg    int
}

// Base data for handler functions to view input image dataset
func NewImagePage(t *Templates, net *Network, scale float64, rows, cols int) *ImagePage {
	p := &ImagePage{net: net, Dset: "train"}
	p.Templates = t.Select("images")
	p.AddOption(Link{Name: "all"})
	p.AddOption(Link{Name: "errors"})
	p.AddOption(Link{Name: "prev"})
	p.AddOption(Link{Name: "next"})
	p.AddOption(Link{Name: "distort"})
	for _, key := range nnet.DataTypes {
		if _, ok := net.Data[key]; ok {
			p.AddOption(Link{Name: key, Url: "/images/all/" + key + "/1", Selected: key == p.Dset})
		}
	}
	dims := net.Data["train"].Shape()
	p.Width = int(float64(dims[1]) * scale)
	p.Height = int(float64(dims[0]) * scale)
	p.Rows = seq(rows)
	p.Cols = seq(cols)
	return p
}

// Handler function for the image grid
func (p *ImagePage) Base() func(w http.ResponseWriter, r *http.Request) {
	return func(w http.ResponseWriter, r *http.Request) {
		p.net.Lock()
		defer p.net.Unlock()
		vars := mux.Vars(r)
		inc := vars["inc"]
		p.errors = (inc == "errors")
		p.Dset = vars["dset"]
		p.page, _ = strconv.Atoi(vars["page"])
		p.Distort = r.FormValue("d")
		//log.Printf("imageBase: %s errors=%v dset=%s page=%d", r.URL.Path, p.errors, p.Dset, p.page)
		if _, ok := p.net.Data[p.Dset]; !ok {
			http.NotFound(w, r)
			return
		}
		p.setPageCount()
		p.Heading = fmt.Sprintf("page %d of %d", p.page, p.pages)
		p.Options[0].Url = "/images/all/" + p.Dset + "/1"
		p.Options[1].Url = "/images/errors/" + p.Dset + "/1"
		p.Options[2].Url = fmt.Sprintf("/images/%s/%s/%d", inc, p.Dset, mod(p.page-1, 1, p.pages))
		p.Options[3].Url = fmt.Sprintf("/images/%s/%s/%d", inc, p.Dset, mod(p.page+1, 1, p.pages))
		p.Options[4].Url = fmt.Sprintf("/images/%s/%s/%d", inc, p.Dset, p.page)
		if p.Distort == "" {
			p.Options[4].Url += "?d=" + strconv.Itoa(rand.Intn(999999))
		} else {
			p.Options[4].Selected = true
		}
		p.SelectOption(inc, p.Dset)
		if err := p.ExecuteTemplate(w, "images", p); err != nil {
			logError(w, err)
		}
	}
}

func (p *ImagePage) setPageCount() {
	if p.errors {
		p.nimg = p.errorImageCount(-1)
	} else {
		p.nimg = p.net.Data[p.Dset].Len()
	}
	rows, cols := len(p.Rows), len(p.Cols)
	p.pages = p.nimg / (rows * cols)
	if p.nimg%(rows*cols) != 0 {
		p.pages++
	}
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
	index := (p.page-1)*rows*cols + row*cols + col
	if index >= p.nimg {
		return 0
	}
	if p.errors {
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
		image = img.Highlight(image, p.Label(id) != p.Predict(id))
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
