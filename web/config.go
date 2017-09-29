package web

import (
	"fmt"
	"github.com/jnb666/deepthought2/nnet"
	"html/template"
	"io/ioutil"
	"log"
	"net/http"
	"strings"
	"sync"
)

type ConfigPage struct {
	*Templates
	Fields []Field
	Layers []Layer
	conf   *Config
	sync.Mutex
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
func NewConfigPage(t *Templates, conf *Config) *ConfigPage {
	p := &ConfigPage{conf: conf}
	p.Templates = t.Select("config")
	p.AddOption(Link{Name: "save", Url: "/config/save", Submit: true})
	p.AddOption(Link{Name: "reset", Url: "/config/reset"})
	p.Fields = getFields(&conf.Config)
	p.Layers = getLayers(&conf.Config)
	return p
}

// Handler function for the config template
func (p *ConfigPage) Base() func(w http.ResponseWriter, r *http.Request) {
	return func(w http.ResponseWriter, r *http.Request) {
		p.Lock()
		defer p.Unlock()
		//log.Println("configBase:", r.URL.Path, r.Method)
		if err := p.ExecuteTemplate(w, "config", p); err != nil {
			logError(w, err)
		}
	}
}

// Handler function for the action to load a new model
func (p *ConfigPage) Load() func(w http.ResponseWriter, r *http.Request) {
	return func(w http.ResponseWriter, r *http.Request) {
		p.Lock()
		defer p.Unlock()
		//log.Println("configLoad:", r.URL.Path, r.Method)
		model := r.FormValue("model")
		log.Println("load model:", model)
		conf, err := nnet.LoadConfig(model + ".net")
		if err != nil {
			logError(w, err)
			return
		}
		p.conf.Config = conf
		p.conf.Model = model
		p.Fields = getFields(&p.conf.Config)
		p.Layers = getLayers(&p.conf.Config)
		http.Redirect(w, r, "/config", http.StatusFound)
	}
}

// Handler function for the config form save action
func (p *ConfigPage) Save() func(w http.ResponseWriter, r *http.Request) {
	return func(w http.ResponseWriter, r *http.Request) {
		p.Lock()
		defer p.Unlock()
		//log.Println("configSave:", r.URL.Path, r.Method)
		r.ParseForm()
		haveErrors := false
		conf := p.conf.Config
		for i, fld := range p.Fields {
			val := r.Form.Get(fld.Name)
			var err error
			if fld.Boolean {
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
			if err := conf.Save(p.conf.Model + ".net"); err != nil {
				logError(w, err)
				return
			}
			p.conf.Config = conf
		}
		http.Redirect(w, r, "/config", http.StatusFound)
	}
}

// Handler function for the config form save action
func (p *ConfigPage) Reset() func(w http.ResponseWriter, r *http.Request) {
	return func(w http.ResponseWriter, r *http.Request) {
		p.Lock()
		defer p.Unlock()
		//log.Println("configReset:", r.URL.Path, r.Method)
		conf, err := nnet.LoadConfig(p.conf.Model + ".default")
		if err != nil {
			logError(w, err)
			return
		}
		if err = conf.Save(p.conf.Model + ".net"); err != nil {
			logError(w, err)
			return
		}
		p.conf.Config = conf
		p.Fields = getFields(&conf)
		http.Redirect(w, r, "/config", http.StatusFound)
	}
}

func (p *ConfigPage) Heading() template.HTML {
	files, err := ioutil.ReadDir(nnet.DataDir)
	if err != nil {
		log.Fatal(err)
	}
	html := `model: <select name="model" class="model-select" form="loadConfig" onchange="this.form.submit()">`
	for _, file := range files {
		name := file.Name()
		if strings.HasSuffix(name, ".net") {
			name = name[:len(name)-4]
			if name == p.conf.Model {
				html += "<option selected>" + name + "</option>"
			} else {
				html += "<option>" + name + "</option>"
			}
		}
	}
	html += "</select>"
	return template.HTML(html)
}

func getFields(conf *nnet.Config) []Field {
	keys := conf.Fields()
	var flds []Field
	for _, key := range keys {
		if key != "UseGPU" {
			f := Field{Name: key, Value: fmt.Sprint(conf.Get(key))}
			f.On, f.Boolean = conf.Get(key).(bool)
			flds = append(flds, f)
		}
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
