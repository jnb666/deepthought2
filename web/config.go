package web

import (
	"fmt"
	"github.com/jnb666/deepthought2/nnet"
	"net/http"
	"reflect"
	"strconv"
	"sync"
)

type ConfigPage struct {
	*Templates
	Heading string
	Fields  []Field
	Layers  []Layer
	model   string
	conf    nnet.Config
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
func NewConfigPage(t *Templates, model string, conf nnet.Config) *ConfigPage {
	p := &ConfigPage{model: model, conf: conf}
	p.Templates = t.Select("config")
	p.AddOption(Link{Name: "save", Url: "/config/save", Submit: true})
	p.AddOption(Link{Name: "reset", Url: "/config/reset"})
	p.Heading = "model: " + model
	p.Fields = getFields(&conf)
	p.Layers = getLayers(&conf)
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

// Handler function for the config form save action
func (p *ConfigPage) Save() func(w http.ResponseWriter, r *http.Request) {
	return func(w http.ResponseWriter, r *http.Request) {
		p.Lock()
		defer p.Unlock()
		//log.Println("configSave:", r.URL.Path, r.Method)
		r.ParseForm()
		s := reflect.ValueOf(&p.conf).Elem()
		haveErrors := false
		for i, fld := range p.Fields {
			val := r.Form.Get(fld.Name)
			p.Fields[i].Value = val
			f := s.FieldByName(fld.Name)
			var err error
			switch f.Type().Kind() {
			case reflect.Int, reflect.Int64:
				var x int64
				if x, err = strconv.ParseInt(val, 10, 64); err == nil {
					f.SetInt(x)
				}
			case reflect.Float64:
				var x float64
				if x, err = strconv.ParseFloat(val, 64); err == nil {
					f.SetFloat(x)
				}
			case reflect.Bool:
				p.Fields[i].On = (val == "true")
				f.SetBool(p.Fields[i].On)
			}
			//log.Println(fld.Name, "=>", val, err)
			p.Fields[i].Error = ""
			if err != nil {
				p.Fields[i].Error = "invalid syntax"
				haveErrors = true
			}
		}
		if !haveErrors {
			if err := p.conf.Save(p.model); err != nil {
				logError(w, err)
				return
			}
		}
		http.Redirect(w, r, "/config", http.StatusFound)
	}
}

// Handler function for the config form save action
func (p *ConfigPage) Reset() func(w http.ResponseWriter, r *http.Request) {
	return func(w http.ResponseWriter, r *http.Request) {
		p.Lock()
		defer p.Unlock()
		var err error
		//log.Println("configReset:", r.URL.Path, r.Method)
		if p.conf, err = nnet.LoadConfig(p.model + "_default"); err != nil {
			logError(w, err)
			return
		}
		if err = p.conf.Save(p.model); err != nil {
			logError(w, err)
			return
		}
		p.Fields = getFields(&p.conf)
		http.Redirect(w, r, "/config", http.StatusFound)
	}
}

func getFields(conf *nnet.Config) []Field {
	keys := conf.Fields()
	flds := make([]Field, len(keys))
	for i, key := range keys {
		if flag, ok := conf.Get(key).(bool); ok {
			flds[i].Boolean = true
			flds[i].On = flag
		}
		flds[i].Name = key
		flds[i].Value = fmt.Sprint(conf.Get(key))
	}
	return flds
}

func getLayers(conf *nnet.Config) []Layer {
	layers := make([]Layer, len(conf.Layers))
	for i, l := range conf.Layers {
		layers[i].Index = i
		layers[i].Desc = l.Unmarshal().String()
	}
	return layers
}
