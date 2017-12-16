package web

import (
	"fmt"
	"github.com/gorilla/mux"
	"github.com/gorilla/sessions"
	"github.com/jnb666/deepthought2/img"
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
	p.Templates = t.Select("/images/train")
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

// Handler function for the image grid
func (p *ImagePage) Base() func(w http.ResponseWriter, r *http.Request) {
	return func(w http.ResponseWriter, r *http.Request) {
		p.net.Lock()
		defer p.net.Unlock()
		session, err := p.getSession(r)
		if err != nil {
			logError(w, err)
			return
		}
		vars := mux.Vars(r)
		p.Dset = vars["dset"]
		//log.Printf("imageBase: %s errors=%v dset=%s page=%d distort=%s", r.URL.Path, p.errors, p.Dset, p.page, p.Distort)
		p.Select("/images/" + p.Dset)
		sel := []string{"all"}
		if p.errors {
			sel = []string{"errors"}
		}
		if p.Distort != "" {
			sel = append(sel, "distort")
		}
		p.SelectOptions(sel)
		p.setPageCount()
		if p.page > p.pages || p.page < 1 {
			p.page = 1
		}
		p.Heading = fmt.Sprintf("page %d of %d", p.page, p.pages)
		template := "images"
		if _, ok := p.net.Data[p.Dset]; !ok {
			template = "blank"
		}
		if err := p.saveSession(r, w, session); err != nil {
			logError(w, err)
			return
		}
		if err := p.ExecuteTemplate(w, template, p); err != nil {
			logError(w, err)
		}
	}
}

// Set option from top menu
func (p *ImagePage) Setopt() func(w http.ResponseWriter, r *http.Request) {
	return func(w http.ResponseWriter, r *http.Request) {
		p.net.Lock()
		defer p.net.Unlock()
		session, err := p.getSession(r)
		if err != nil {
			logError(w, err)
			return
		}
		vars := mux.Vars(r)
		p.Dset = vars["dset"]
		p.setPageCount()
		switch vars["opt"] {
		case "all":
			p.errors = false
		case "errors":
			p.errors = true
		case "prev":
			p.page = mod(p.page-1, 1, p.pages)
		case "next":
			p.page = mod(p.page+1, 1, p.pages)
		case "distort":
			if p.Distort == "" {
				p.Distort = strconv.Itoa(rand.Intn(999999))
			} else {
				p.Distort = ""
			}
		}
		if err := p.saveSession(r, w, session); err != nil {
			logError(w, err)
			return
		}
		http.Redirect(w, r, "/images/"+p.Dset+"/", http.StatusFound)
	}
}

func (p *ImagePage) getSession(r *http.Request) (*sessions.Session, error) {
	s, err := p.store.Get(r, "deepthought")
	if err != nil {
		return nil, err
	}
	var ok bool
	if p.page, ok = s.Values["page"].(int); !ok {
		s.Values["page"] = 1
	}
	if p.errors, ok = s.Values["errors"].(bool); !ok {
		s.Values["errors"] = false
	}
	if p.Distort = s.Values["distort"].(string); !ok {
		s.Values["distort"] = ""
	}
	return s, nil
}

func (p *ImagePage) saveSession(r *http.Request, w http.ResponseWriter, s *sessions.Session) error {
	s.Values["page"] = p.page
	s.Values["errors"] = p.errors
	s.Values["distort"] = p.Distort
	return s.Save(r, w)
}

func (p *ImagePage) setPageCount() {
	if _, ok := p.net.Data[p.Dset]; !ok {
		p.pages = 1
		return
	}
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
