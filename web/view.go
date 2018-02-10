package web

import (
	"fmt"
	"github.com/gorilla/mux"
	"html/template"
	"image"
	"image/color"
	"image/png"
	"net/http"
	"strconv"
	"time"
)

type ViewPage struct {
	*Templates
	Page    string
	Input   LayerInfo
	Output  LayerInfo
	Layers  []LayerInfo
	Columns int
	net     *Network
	index   int
	errors  bool
}

type LayerInfo struct {
	Desc      string
	Image     []string
	Values    []template.HTML
	Cols      int
	Width     int
	CellWidth int
	PadWidth  int
	Class     string
}

// Base data for handler functions to view network activations and weights
func NewViewPage(t *Templates, net *Network) *ViewPage {
	p := &ViewPage{net: net, Templates: t, index: 1}
	for _, name := range []string{"all", "errors", "prev", "next"} {
		p.AddOption(Link{Name: name, Url: "./" + name})
	}
	p.SelectOptions([]string{"all"})
	return p
}

// Handler function for the main view page
func (p *ViewPage) Base() func(w http.ResponseWriter, r *http.Request) {
	return func(w http.ResponseWriter, r *http.Request) {
		p.net.Lock()
		defer p.net.Unlock()
		vars := mux.Vars(r)
		p.Page = vars["page"]
		p.Select("/view/" + p.Page + "/")
		if p.errors {
			p.SelectOptions([]string{"errors"})
		} else {
			p.SelectOptions([]string{"all"})
		}
		p.Heading = p.net.heading()
		p.Frame = "/net/" + p.Page
		p.Exec(w, "frame", p, true)
	}
}

// Set option from top menu
func (p *ViewPage) Setopt() func(w http.ResponseWriter, r *http.Request) {
	return func(w http.ResponseWriter, r *http.Request) {
		p.net.Lock()
		defer p.net.Unlock()
		nimg := p.net.view.data.Len()
		if p.errors {
			pred := p.net.Pred[p.net.view.dset]
			for i, label := range p.net.Labels[p.net.view.dset] {
				if i < len(pred) && pred[i] != label {
					nimg++
				}
			}
		}
		vars := mux.Vars(r)
		switch vars["opt"] {
		case "all":
			p.errors = false
		case "errors":
			p.errors = true
		case "prev":
			p.index = mod(p.index-1, 1, nimg)
		case "next":
			p.index = mod(p.index+1, 1, nimg)
		}
		http.Redirect(w, r, "/view/"+vars["page"]+"/", http.StatusFound)
	}
}

// Handler function for the frame with activation or weights visualisation
func (p *ViewPage) Network() func(w http.ResponseWriter, r *http.Request) {
	return func(w http.ResponseWriter, r *http.Request) {
		p.net.Lock()
		defer p.net.Unlock()
		vars := mux.Vars(r)
		p.Page = vars["page"]
		if p.index > p.net.view.data.Len() {
			p.index = 1
		}
		index := p.index - 1
		if p.errors {
			pred := p.net.Pred[p.net.view.dset]
			nimg := 0
			index = 0
			for i, label := range p.net.Labels[p.net.view.dset] {
				if i < len(pred) && pred[i] != label {
					nimg++
				}
				if nimg == p.index {
					index = i
					break
				}
			}
		}
		p.getLayers(index)
		p.Exec(w, "net", p, false)
	}
}

// Used in template the display the layer data
func (p *ViewPage) getLayers(index int) {
	p.Layers = []LayerInfo{}
	ts := time.Now().Unix()
	switch p.Page {
	case "outputs":
		p.Columns = 2
		p.net.view.updateOutputs(index)
		// input image
		classes := p.net.view.data.Classes()
		label := p.net.Labels[p.net.view.dset][index]
		p.Input = info(
			fmt.Sprintf("input %d %v => %s", index+1, p.net.view.inShape, classes[label]),
			fmt.Sprintf("/img/%s/%d/", p.net.view.dset, index+1),
			p.net.view.data.Image(index, ""),
			len(p.net.view.inShape) == 3 && p.net.view.inShape[2] == 3,
			0, 0, "input",
		)
		// outputs at each layer
		for i, l := range p.net.view.layers {
			entry := info(l.desc,
				fmt.Sprintf("/net/outputs/%d?ts=%d", i, ts),
				l.image, false, factorMinOutput, 100, "weights",
			)
			if i == len(p.net.view.layers)-1 {
				p.Output = entry
			} else {
				p.Layers = append(p.Layers, entry)
			}
		}
		// output classification
		if l := p.net.view.lastLayer(); l != nil {
			if len(l.data) > 1 {
				imax := arrayMax(l.data)
				width := int(100 / float64(len(l.data)))
				for i, val := range l.data {
					p.addOutput(val, classes[i], i == imax, width)
				}
			} else {
				class := classes[0]
				if l.data[0] > 0.5 {
					class = classes[1]
				}
				p.addOutput(l.data[0], class, false, 100)
			}
		}
	case "weights":
		p.Columns = 1
		p.net.view.updateWeights()
		// display weights and biases
		for i, l := range p.net.view.weights {
			entry := info(l.desc, fmt.Sprintf("/net/weights/%d?ts=%d", i, ts),
				l.image, false, factorMinWeights, 100, "weights",
			)
			p.Layers = append(p.Layers, entry)
		}
	}
}

func (p *ViewPage) addOutput(val float32, class string, underline bool, width int) {
	col := color.Gray{Y: uint8(255 - 128*(1-val))}
	tag := fmt.Sprintf(`<span style="color:%s;">%s</span>`, htmlColor(col), class)
	if underline {
		tag = "<u>" + tag + "</u>"
	}
	p.Output.Class = "outputs"
	p.Output.Width = 100
	p.Output.PadWidth = 0
	p.Output.Values = append(p.Output.Values, template.HTML(tag))
	p.Output.CellWidth = width
}

func info(desc, url string, img image.Image, channels bool, factorMin, scaleWidth int, class string) LayerInfo {
	info := LayerInfo{Desc: desc, Width: 100, Class: class, Image: []string{}}
	if img != nil {
		info.Image = []string{url}
		if class == "input" {
			info.Width = 20
			if channels {
				for _, suffix := range []string{"r", "g", "b"} {
					info.Image = append(info.Image, url+suffix)
				}
			}
		} else {
			if img.Bounds().Dx() <= scaleWidth {
				info.Width /= 2
			}
		}
	}
	info.PadWidth = 100 - len(info.Image)*info.Width
	info.Cols = len(info.Image) + 1
	return info
}

// Handler function to generate the image for the output and weight data visualisation
func (p *ViewPage) Image() func(w http.ResponseWriter, r *http.Request) {
	return func(w http.ResponseWriter, r *http.Request) {
		p.net.Lock()
		defer p.net.Unlock()
		vars := mux.Vars(r)
		page := vars["page"]
		layer, _ := strconv.Atoi(vars["layer"])
		var img *image.NRGBA
		switch page {
		case "outputs":
			if layer >= 0 && layer < len(p.net.view.layers) {
				img = p.net.view.layers[layer].image
			}
		case "weights":
			if layer >= 0 && layer < len(p.net.view.weights) {
				img = p.net.view.weights[layer].image
			}
		}
		if img == nil {
			http.NotFound(w, r)
			return
		}
		w.Header().Set("Content-type", "image/png")
		png.Encode(w, img)
	}
}

func arrayMax(a []float32) int {
	max := a[0]
	imax := 0
	for i, val := range a[1:] {
		if val > max {
			max = val
			imax = i + 1
		}
	}
	return imax
}
