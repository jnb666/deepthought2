package web

import (
	"bytes"
	"fmt"
	"github.com/gorilla/mux"
	"html/template"
	"image"
	"image/color"
	"image/png"
	"log"
	"net/http"
	"strconv"
	"time"
)

type ViewPage struct {
	*Templates
	Page   string
	Layers []LayerInfo
	net    *Network
	index  int
	errors bool
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
	ImageData bytes.Buffer
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
		p.Exec(w, "view", p, true)
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
		p.net.view.updateOutputs(index)
		// input image
		classes := p.net.view.data.Classes()
		label := p.net.Labels[p.net.view.dset][index]
		norm := ""
		if p.net.Network.Normalise {
			norm = "n"
		}
		p.addImage(
			fmt.Sprintf("input %d %v => %s", index+1, p.net.view.inShape, classes[label]),
			fmt.Sprintf("/img/%s/%d/%s", p.net.view.dset, index+1, norm),
			p.net.view.data.Image(index, ""),
			len(p.net.view.inShape) == 3 && p.net.view.inShape[2] == 3,
			0, 0, "input",
		)
		// outputs at each layer
		for i, l := range p.net.view.layers {
			if l.outShape != nil {
				p.addImage(
					fmt.Sprintf("%d: %s %v", i, l.ltype, l.outShape),
					fmt.Sprintf("/net/outputs/%d?ts=%d", len(p.Layers), ts),
					l.outImage, false, factorMinOutput, 100, "weights",
				)
			}
		}
		if l := p.net.view.lastLayer(); l != nil {
			if len(l.outData) > 1 {
				imax := arrayMax(l.outData)
				width := int(100 / float64(len(l.outData)))
				for i, val := range l.outData {
					p.addOutput(val, classes[i], i == imax, width)
				}
			} else {
				class := classes[0]
				if l.outData[0] > 0.5 {
					class = classes[1]
				}
				p.addOutput(l.outData[0], class, false, 100)
			}
		}
	case "weights":
		p.net.view.updateWeights()
		// display weights and biases
		for i, l := range p.net.view.layers {
			if l.wShape != nil {
				p.addImage(
					fmt.Sprintf("%d: %s %v %v", i, l.ltype, l.wShape, l.bShape),
					fmt.Sprintf("/net/weights/%d?ts=%d", len(p.Layers), ts),
					l.wImage, false, factorMinWeights, 100, "weights",
				)
			}
		}
	}
}

func (p *ViewPage) addOutput(val float32, class string, underline bool, width int) {
	out := len(p.Layers) - 1
	col := color.Gray{Y: uint8(255 - 128*(1-val))}
	tag := fmt.Sprintf(`<span style="color:%s;">%s</span>`, htmlColor(col), class)
	if underline {
		tag = "<u>" + tag + "</u>"
	}
	p.Layers[out].Class = "outputs"
	p.Layers[out].Values = append(p.Layers[out].Values, template.HTML(tag))
	p.Layers[out].CellWidth = width
}

func (p *ViewPage) addImage(desc, url string, img image.Image, channels bool, factorMin, scaleWidth int, class string) {
	info := LayerInfo{Desc: desc, Width: 100, Class: class, Image: []string{}}
	if img != nil {
		info.Image = []string{url}
		if class == "input" {
			info.Width = 7
			if channels {
				for _, suffix := range []string{"r", "g", "b"} {
					info.Image = append(info.Image, url+suffix)
				}
			}
		} else {
			if img.Bounds().Dx() <= scaleWidth {
				info.Width /= 2
			}
			if err := png.Encode(&info.ImageData, img); err != nil {
				log.Println(err)
			}
		}
	}
	info.PadWidth = 100 - len(info.Image)*info.Width
	info.Cols = len(info.Image) + 1
	p.Layers = append(p.Layers, info)
}

// Handler function to generate the image for the output and weight data visualisation
func (p *ViewPage) Image() func(w http.ResponseWriter, r *http.Request) {
	return func(w http.ResponseWriter, r *http.Request) {
		p.net.Lock()
		defer p.net.Unlock()
		vars := mux.Vars(r)
		page := vars["page"]
		layer, _ := strconv.Atoi(vars["layer"])
		if layer >= len(p.Layers) || p.Layers[layer].ImageData.Len() == 0 {
			log.Printf("viewImage: not found page=%s layer=%d\n", page, layer)
			http.NotFound(w, r)
			return
		}
		w.Header().Set("Content-type", "image/png")
		w.Write(p.Layers[layer].ImageData.Bytes())
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
