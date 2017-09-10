package web

import (
	"fmt"
	"html/template"
	"log"
	"net/http"
	"os"
)

var AssetDir = os.Getenv("GOPATH") + "/src/github.com/jnb666/deepthought2/assets"

// Template and main menu definition
type Templates struct {
	*template.Template
	Menu    []Link
	Options []Link
}

type Link struct {
	Url      string
	Name     string
	Selected bool
	Submit   bool
}

// Load and parse templates and initialise main menu
func NewTemplates() (*Templates, error) {
	var err error
	t := &Templates{Options: []Link{}}
	t.Template, err = template.ParseGlob(AssetDir + "/*.html")
	if err != nil {
		return nil, err
	}
	for _, name := range []string{"train", "images", "config"} {
		t.Menu = append(t.Menu, Link{Name: name, Url: "/" + name})
	}
	return t, nil
}

func (t *Templates) Clone() *Templates {
	return &Templates{
		Template: t.Template,
		Menu:     append([]Link{}, t.Menu...),
		Options:  append([]Link{}, t.Options...),
	}
}

func (t *Templates) Select(name string) *Templates {
	for i, key := range t.Menu {
		t.Menu[i].Selected = (key.Name == name)
	}
	return t
}

func (t *Templates) AddOption(l Link) *Templates {
	t.Options = append(t.Options, l)
	return t
}

func (t *Templates) SelectOption(names ...string) *Templates {
	for i, key := range t.Options {
		t.Options[i].Selected = false
		for _, name := range names {
			if key.Name == name {
				t.Options[i].Selected = true
			}
		}
	}
	return t
}

func logError(w http.ResponseWriter, err error) {
	log.Println(err)
	http.Error(w, fmt.Sprint(err), http.StatusInternalServerError)
}
