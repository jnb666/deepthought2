package web

import (
	"errors"
	"fmt"
	"github.com/jnb666/deepthought2/nnet"
	"html/template"
	"io/ioutil"
	"log"
	"net/http"
	"sort"
	"strings"
)

var ErrMissingField = errors.New("value is required")

type ConfigPage struct {
	*Templates
	Fields []Field
	Layers []Layer
	net    *Network
}

type Field struct {
	Name    string
	Value   string
	Error   string
	Boolean bool
	On      bool
}

type Layer struct {
	Index int
	Desc  string
}

// Base data for handler functions to view and update the network config
func NewConfigPage(t *Templates, net *Network) *ConfigPage {
	p := &ConfigPage{net: net, Templates: t}
	p.AddOption(Link{Name: "save", Url: "/config/save", Submit: true})
	p.AddOption(Link{Name: "reset", Url: "/config/reset"})
	p.init(nil)
	return p
}

func (p *ConfigPage) init(data *NetworkData) error {
	if data != nil {
		p.net.NetworkData = data
		if err := p.net.Init(); err != nil {
			return err
		}
		if err := p.net.Import(); err != nil {
			return err
		}
	}
	p.Fields = getFields(&p.net.Conf, p.net.Model)
	p.Layers = getLayers(&p.net.Conf)
	return nil
}

// Handler function for the config template
func (p *ConfigPage) Base() func(w http.ResponseWriter, r *http.Request) {
	return func(w http.ResponseWriter, r *http.Request) {
		p.net.Lock()
		defer p.net.Unlock()
		p.Select("/config")
		p.Heading = p.getHeading()
		p.Exec(w, "config", p)
	}
}

// Handler function for the action to load a new model
func (p *ConfigPage) Load() func(w http.ResponseWriter, r *http.Request) {
	return func(w http.ResponseWriter, r *http.Request) {
		p.net.Lock()
		defer p.net.Unlock()
		model := r.FormValue("model")
		data, err := LoadNetwork(model, false)
		if err != nil {
			p.logError(w, http.StatusBadRequest, err)
			return
		}
		if err = p.init(data); err != nil {
			p.logError(w, http.StatusInternalServerError, err)
			return
		}
		http.Redirect(w, r, "/config", http.StatusFound)
	}
}

// Handler function for the config form save action
func (p *ConfigPage) Save() func(w http.ResponseWriter, r *http.Request) {
	return func(w http.ResponseWriter, r *http.Request) {
		p.net.Lock()
		defer p.net.Unlock()
		r.ParseForm()
		haveErrors := false
		conf := p.net.Conf
		model := p.net.Model
		for i, fld := range p.Fields {
			val := r.Form.Get(fld.Name)
			var err error
			if fld.Name == "Model" {
				if val != "" {
					p.Fields[i].Value = val
					model = val
				} else {
					err = ErrMissingField
				}
			} else if fld.Boolean {
				p.Fields[i].On = (val == "true")
				conf, err = conf.SetBool(fld.Name, p.Fields[i].On)
			} else {
				p.Fields[i].Value = val
				conf, err = conf.SetString(fld.Name, val)
			}
			p.Fields[i].Error = ""
			if err != nil {
				p.Fields[i].Error = "invalid syntax"
				haveErrors = true
			}
		}
		if !haveErrors {
			p.net.Model = model
			p.net.Conf = conf
			p.net.Export()
			p.net.updated = true
			if err := SaveNetwork(p.net.NetworkData); err != nil {
				p.logError(w, http.StatusBadRequest, err)
				return
			}
		}
		http.Redirect(w, r, "/config", http.StatusFound)
	}
}

// Handler function for the config form save action
func (p *ConfigPage) Reset() func(w http.ResponseWriter, r *http.Request) {
	return func(w http.ResponseWriter, r *http.Request) {
		p.net.Lock()
		defer p.net.Unlock()
		data, err := LoadNetwork(p.net.Model, true)
		if err != nil {
			p.logError(w, http.StatusBadRequest, err)
			return
		}
		if err = p.init(data); err != nil {
			p.logError(w, http.StatusInternalServerError, err)
			return
		}
		if err := SaveNetwork(data); err != nil {
			p.logError(w, http.StatusBadRequest, err)
			return
		}
		http.Redirect(w, r, "/config", http.StatusFound)
	}
}

func (p *ConfigPage) getHeading() template.HTML {
	files, err := ioutil.ReadDir(nnet.DataDir)
	if err != nil {
		log.Println("Error reading DataDir:", err)
		return ""
	}
	models := make(map[string]bool)
	for _, file := range files {
		name := file.Name()
		for _, suffix := range []string{".conf", ".net"} {
			if strings.HasSuffix(name, suffix) {
				models[name[:len(name)-len(suffix)]] = true
			}
		}
	}
	names := make([]string, 0, len(models))
	for name := range models {
		names = append(names, name)
	}
	sort.Strings(names)
	html := `model: <select name="model" class="model-select" form="loadConfig" onchange="this.form.submit()">`
	for _, name := range names {
		if name == p.net.Model {
			html += "<option selected>" + name + "</option>"
		} else {
			html += "<option>" + name + "</option>"
		}
	}
	html += "</select>"
	return template.HTML(html)
}

func getFields(conf *nnet.Config, model string) []Field {
	flds := []Field{{Name: "Model", Value: model}}
	keys := conf.Fields()
	for _, key := range keys {
		f := Field{Name: key, Value: fmt.Sprint(conf.Get(key))}
		f.On, f.Boolean = conf.Get(key).(bool)
		flds = append(flds, f)
	}
	return flds
}

func getLayers(conf *nnet.Config) []Layer {
	layers := make([]Layer, len(conf.Layers))
	for i, l := range conf.Layers {
		layers[i].Index = i
		layers[i].Desc = l.Unmarshal().ToString()
	}
	return layers
}
