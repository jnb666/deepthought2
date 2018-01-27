package web

import (
	"fmt"
	"github.com/gorilla/mux"
	"github.com/jnb666/deepthought2/img"
	"image/png"
	"log"
	"math/rand"
	"net/http"
	"strconv"
)

type ImagePage struct {
	*Templates
	Dset    string
	Class   int
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
	height  int
	width   int
}

type LabelInfo struct {
	Desc  string
	Pred  string
	Class string
}

// Base data for handler functions to view input image dataset
func NewImagePage(t *Templates, net *Network, rows, cols, height, width int) *ImagePage {
	p := &ImagePage{net: net,
		Templates: t,
		Page:      1,
		Rows:      seq(rows),
		Cols:      seq(cols),
		height:    height,
		width:     width,
	}
	for _, name := range []string{"errors", "prev", "next", "distort", "normalise"} {
		p.AddOption(Link{Name: name, Url: "./" + name})
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
		if vars["class"] != "" {
			p.Class, _ = strconv.Atoi(vars["class"])
		}
		base := "/images/" + p.Dset + "/"
		p.Select(base)
		p.Heading = p.net.heading()
		template := "blank"
		if d, ok := p.net.Data[p.Dset]; ok {
			template = "frame"
			p.Frame = "/grid/" + p.Dset
			p.Dropdown = []Link{{Name: "all classes", Url: base + "0"}}
			for i, class := range d.Classes() {
				p.Dropdown = append(p.Dropdown, Link{Name: class, Url: base + strconv.Itoa(i+1), Selected: i+1 == p.Class})
			}
			dims := d.Shape()
			h, w := dims[0], 1
			scale := int((((float64(p.height) - 100) / float64(len(p.Rows))) - 18)) / h
			if len(dims) >= 2 {
				w = dims[1]
				scalex := int((((float64(p.width) - 32) / float64(len(p.Cols))) - 4)) / w
				if scalex < scale {
					scale = scalex
				}
			}
			p.Width = w * scale
			p.Height = h * scale
		} else {
			p.Dropdown = nil
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
		if _, ok := p.net.Data[p.Dset]; !ok {
			p.logError(w, http.StatusNotFound, ErrNotFound)
			return
		}
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
		case "errors":
			p.Errors = p.ToggleOption("errors")
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
			p.ToggleOption("distort")
		case "normalise":
			p.ToggleOption("normalise")
		}
		http.Redirect(w, r, "/images/"+p.Dset+"/", http.StatusFound)
	}
}

func (p *ImagePage) pageCount() (nimg, pages int) {
	for i := range p.net.Labels[p.Dset] {
		if p.showImage(i) {
			nimg++
		}
	}
	rows, cols := len(p.Rows), len(p.Cols)
	pages = nimg / (rows * cols)
	if nimg%(rows*cols) != 0 {
		pages++
	}
	return nimg, pages
}

func (p *ImagePage) showImage(i int) bool {
	labels := p.net.Labels[p.Dset]
	if i >= len(labels) {
		return false
	}
	show := p.Class == 0 || int(labels[i]) == p.Class-1
	if p.Errors {
		if pred, ok := p.net.Pred[p.Dset]; ok {
			show = show && pred[i] != labels[i]
		} else {
			show = false
		}
	}
	return show
}

func (p *ImagePage) Index(row, col int) int {
	rows, cols := len(p.Rows), len(p.Cols)
	index := (p.Page-1)*rows*cols + row*cols + col
	for i := range p.net.Labels[p.Dset] {
		if p.showImage(i) {
			index--
			if index < 0 {
				return i + 1
			}
		}
	}
	return 0
}

func (p *ImagePage) Url(i int) string {
	url := strconv.Itoa(i) + "/"
	if p.OptionSelected("normalise") {
		url += "n"
	}
	if p.Distort != "" {
		url += "?d=" + p.Distort
	}
	return url
}

func (p *ImagePage) Label(i int) (l LabelInfo) {
	i--
	l.Class = "image-ok"
	data := p.net.Data[p.Dset]
	labels := p.net.Labels[p.Dset]
	predict := p.net.Pred[p.Dset]
	if data == nil || labels == nil || i < 0 || i >= len(labels) {
		return
	}
	l.Desc = fmt.Sprintf("%d: %s", i+1, data.Classes()[labels[i]])
	if predict == nil || i >= len(predict) {
		return
	}
	l.Pred = fmt.Sprint(data.Classes()[predict[i]])
	if predict[i] != labels[i] {
		l.Class = "image-error"
	}
	return
}

// Handler function for the image data
func (p *ImagePage) Image() func(w http.ResponseWriter, r *http.Request) {
	return func(w http.ResponseWriter, r *http.Request) {
		p.net.Lock()
		defer p.net.Unlock()
		vars := mux.Vars(r)
		dset := vars["dset"]
		id, _ := strconv.Atoi(vars["id"])
		id--
		data, ok := p.net.Data[dset]
		if !ok || id < 0 || id >= data.Len() {
			http.NotFound(w, r)
			return
		}
		opts := vars["opts"]
		norm := false
		if len(opts) > 0 && opts[0] == 'n' {
			norm = true
			opts = opts[1:]
		}
		m := data.Image(id, opts)
		distort := r.FormValue("d") != ""
		if norm || distort {
			m = p.transform(m, dset, id, norm, distort)
		}
		inError := p.net.Pred[dset] != nil && id < len(p.net.Pred[dset]) && p.net.Pred[dset][id] != p.net.Labels[dset][id]
		m = img.Highlight(m, inError)
		w.Header().Set("Content-type", "image/png")
		png.Encode(w, m)
	}
}

func (p *ImagePage) transform(src *img.Image, dset string, id int, norm, distort bool) *img.Image {
	t := p.net.trans[dset]
	if t == nil {
		return src
	}
	t.Trans = src.TransformType(norm, distort)
	dst, err := t.Transform(src, 0)
	if err != nil {
		log.Println("ERROR: transform error:", err)
	}
	return dst
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
