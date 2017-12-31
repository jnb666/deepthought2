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
	"strconv"
	"strings"
)

var ErrMissingField = errors.New("value is required")

type ConfigPage struct {
	*Templates
	Fields     []Field
	Layers     []Layer
	TuneFields []Field
	net        *Network
}

type Field struct {
	Name    string
	Value   string
	Error   string
	Boolean bool
	Options []string
}

type Layer struct {
	Index int
	Desc  string
	Shape string
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
		if err := p.net.Init(p.net.Conf); err != nil {
			return err
		}
		if err := p.net.Import(); err != nil {
			return err
		}
	}
	p.Fields = getFields(p.net)
	p.Layers = getLayers(p.net)
	p.TuneFields = getTuneFields(p.net.Tuners)
	return nil
}

// Handler function for the config template
func (p *ConfigPage) Base() func(w http.ResponseWriter, r *http.Request) {
	return func(w http.ResponseWriter, r *http.Request) {
		p.net.Lock()
		defer p.net.Unlock()
		p.Select("/config")
		p.Heading = p.getHeading()
		p.Exec(w, "config", p, true)
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
			val := strings.TrimSpace(r.Form.Get(fld.Name))
			var err error
			if fld.Name == "Model" {
				if val != "" {
					p.Fields[i].Value = val
					model = val
				} else {
					err = ErrMissingField
				}
			} else {
				p.Fields[i].Value = val
				if fld.Options != nil {
					val = getOption(fld.Options, val)
				}
				conf, err = conf.SetString(fld.Name, val)
			}
			p.Fields[i].Error = ""
			if err != nil {
				p.Fields[i].Error = "invalid syntax"
				haveErrors = true
			}
		}
		if !haveErrors {
			newModel := model != p.net.Model
			if newModel {
				p.net.Model = model
				p.net.History = p.net.History[:0]
			}
			p.net.Conf = conf
			p.net.Export()
			p.init(nil)
			p.net.updated = true
			if err := SaveNetwork(p.net.NetworkData, newModel); err != nil {
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
		if err := SaveNetwork(data, false); err != nil {
			p.logError(w, http.StatusBadRequest, err)
			return
		}
		http.Redirect(w, r, "/config", http.StatusFound)
	}
}

// Handler function for the tune form
func (p *ConfigPage) Tune() func(w http.ResponseWriter, r *http.Request) {
	return func(w http.ResponseWriter, r *http.Request) {
		p.net.Lock()
		defer p.net.Unlock()
		r.ParseForm()
		haveErrors := false
		vals := make([][]string, len(p.TuneFields))
		var conf nnet.Config
		for i, f := range p.TuneFields {
			sval := r.Form.Get(f.Name)
			vals[i] = strings.Fields(sval)
			var err error
			p.TuneFields[i].Value = sval
			if len(vals[i]) == 0 {
				err = ErrMissingField
			}
			for _, v := range vals[i] {
				if f.Boolean {
					if v != "false" && v != "true" {
						err = ErrMissingField
					}
				} else {
					if _, err = conf.SetString(f.Name, v); err != nil {
						break
					}
				}
			}
			p.TuneFields[i].Error = ""
			if err != nil {
				p.TuneFields[i].Error = "invalid syntax"
				haveErrors = true
			}
		}
		if !haveErrors {
			for i, val := range vals {
				p.net.Tuners[i].Values = val
			}
			p.TuneFields = getTuneFields(p.net.Tuners)
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

func getFields(net *Network) []Field {
	conf := net.Conf
	flds := []Field{{Name: "Model", Value: net.Model}}
	keys := conf.Fields()
	for _, key := range keys {
		f := Field{Name: key}
		switch c := conf.Get(key).(type) {
		default:
			f.Value = fmt.Sprint(c)
		case bool:
			f.Boolean = true
			if c {
				f.Value = "true"
			}
		case nnet.OptionList:
			f.Options = c.Options()
			f.Value = c.String()
		}
		flds = append(flds, f)
	}
	return flds
}

func getLayers(net *Network) []Layer {
	conf := net.Conf
	layers := make([]Layer, len(conf.Layers))
	for i, l := range conf.Layers {
		dims := net.Layers[i].OutShape()
		layers[i].Index = i
		layers[i].Desc = l.Unmarshal().ToString()
		layers[i].Shape = fmt.Sprint(dims[:len(dims)-1])
	}
	return layers
}

func getTuneFields(tuners []TuneParams) []Field {
	flds := make([]Field, len(tuners))
	for i, f := range tuners {
		val := make([]string, len(f.Values))
		for i, v := range f.Values {
			val[i] = fmt.Sprint(v)
		}
		flds[i] = Field{Name: f.Name, Value: strings.Join(val, " ")}
	}
	return flds
}

func getOption(opts []string, val string) string {
	for id, opt := range opts {
		if val == opt {
			return strconv.Itoa(id)
		}
	}
	return ""
}
