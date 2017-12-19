package web

import (
	"fmt"
	"github.com/gorilla/mux"
	"html/template"
	"image"
	"image/png"
	"log"
	"net/http"
	"strconv"
	"time"
)

const scaleWidth = 20

type ViewPage struct {
	*Templates
	Page  string
	Index int
	net   *Network
	info  []LayerInfo
}

type LayerInfo struct {
	Desc     string
	Image    string
	Values   []template.HTML
	Width    int
	PadWidth int
}

// Base data for handler functions to view network activations and weights
func NewViewPage(t *Templates, net *Network) *ViewPage {
	p := &ViewPage{net: net, Templates: t, Index: 1}
	p.AddOption(Link{Name: "prev", Url: "./prev"})
	p.AddOption(Link{Name: "next", Url: "./next"})
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
		p.Heading = p.net.heading()
		p.Toplevel = true
		p.Exec(w, "view", p)
	}
}

// Set option from top menu
func (p *ViewPage) Setopt() func(w http.ResponseWriter, r *http.Request) {
	return func(w http.ResponseWriter, r *http.Request) {
		p.net.Lock()
		defer p.net.Unlock()
		vars := mux.Vars(r)
		switch vars["opt"] {
		case "prev":
			p.Index = mod(p.Index-1, 1, p.net.view.data.Len())
		case "next":
			p.Index = mod(p.Index+1, 1, p.net.view.data.Len())
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
		p.net.view.update(p.Index - 1)
		p.Toplevel = false
		p.Exec(w, "net", p)
	}
}

// Used in template the display the layer data
func (p *ViewPage) Layers() []LayerInfo {
	p.info = []LayerInfo{}
	ts := time.Now().Unix()
	switch p.Page {
	case "outputs":
		// display input and outputs at each layer
		label := p.net.Labels[p.net.view.dset][p.Index-1]
		p.addImage(
			fmt.Sprintf("input %d %v => %d", p.Index, p.net.view.inShape, label),
			fmt.Sprintf("/img/%s/%d", p.net.view.dset, p.Index),
			p.net.view.inImage,
			5,
			0,
		)
		for i, l := range p.net.view.layers {
			if l.outShape != nil {
				p.addImage(
					fmt.Sprintf("%d: %s %v", i, l.ltype, l.outShape),
					fmt.Sprintf("/net/outputs/%d?ts=%d", i, ts),
					l.outImage,
					100,
					factorMinOutput,
				)
			}
		}
		out := len(p.info) - 1
		if l := p.net.view.lastLayer(); l != nil && out >= 0 {
			for i, val := range l.outData {
				v := int(255 * (1 - val))
				tag := fmt.Sprintf(`<span style="color:#%02x%02x%02x;">%d</span>`, v, v, v, i)
				p.info[out].Values = append(p.info[out].Values, template.HTML(tag))
			}

		}
	case "weights":
		// display weights and biases
		for i, l := range p.net.view.layers {
			if l.wShape != nil {
				p.addImage(
					fmt.Sprintf("%d: %s %v %v", i, l.ltype, l.wShape, l.bShape),
					fmt.Sprintf("/net/weights/%d?ts=%d", i, ts),
					l.wImage,
					100,
					factorMinWeights,
				)
			}
		}
	}
	return p.info
}

func (p *ViewPage) addImage(desc, url string, img *image.NRGBA, width, factorMin int) {
	info := LayerInfo{Desc: desc, Width: width}
	if img != nil {
		info.Image = url
		if img.Bounds().Dx() <= scaleWidth {
			info.Width /= 2
		}
	}
	info.PadWidth = 100 - info.Width
	p.info = append(p.info, info)
}

// Handler function to generate the image for the output and weight data visualisation
func (p *ViewPage) Image() func(w http.ResponseWriter, r *http.Request) {
	return func(w http.ResponseWriter, r *http.Request) {
		p.net.Lock()
		defer p.net.Unlock()
		vars := mux.Vars(r)
		page := vars["page"]
		layer, _ := strconv.Atoi(vars["layer"])
		w.Header().Set("Content-type", "image/png")
		if layer < len(p.net.view.layers) {
			l := p.net.view.layers[layer]
			if page == "outputs" && l.outImage != nil {
				png.Encode(w, l.outImage)
				return
			} else if page == "weights" && l.wImage != nil {
				png.Encode(w, l.wImage)
				return
			}
		}
		log.Printf("viewImage: not found page=%s layer=%d\n", page, layer)
		http.NotFound(w, r)
	}
}
