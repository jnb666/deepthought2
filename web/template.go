package web

import (
	"errors"
	"fmt"
	"html/template"
	"log"
	"net/http"
	"os"
)

var AssetDir = os.Getenv("GOPATH") + "/src/github.com/jnb666/deepthought2/assets"

var ErrNotFound = errors.New("page not found")

// Template and main menu definition
type Templates struct {
	*template.Template
	Menu     []Link
	Options  []Link
	Toplevel bool
	Heading  template.HTML
	Error    string
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
	t := &Templates{Menu: []Link{}, Options: []Link{}, Toplevel: true}
	t.Template, err = template.ParseGlob(AssetDir + "/*.html")
	return t, err
}

func (t *Templates) Clone() *Templates {
	temp := *t
	temp.Options = append([]Link{}, t.Options...)
	return &temp
}

func (t *Templates) Select(url string) *Templates {
	for i, key := range t.Menu {
		t.Menu[i].Selected = key.Url == url
	}
	return t
}

func (t *Templates) AddMenuItem(url, name string) *Templates {
	t.Menu = append(t.Menu, Link{Url: url, Name: name})
	return t
}

func (t *Templates) AddOption(l Link) *Templates {
	t.Options = append(t.Options, l)
	return t
}

func (t *Templates) SelectOptions(names []string) *Templates {
	for i, opt := range t.Options {
		t.Options[i].Selected = false
		for _, name := range names {
			if opt.Name == name {
				t.Options[i].Selected = true
			}
		}
	}
	return t
}

func (t *Templates) OptionSelected(name string) bool {
	for _, opt := range t.Options {
		if opt.Name == name {
			return opt.Selected
		}
	}
	return false
}

// Return custom error response
func (t *Templates) ErrorHandler(status int, errorText error) http.Handler {
	return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		t.logError(w, status, errorText)
	})
}

func (t *Templates) logError(w http.ResponseWriter, status int, errorText error) {
	log.Printf("Error %d: %s\n", status, errorText)
	t.Error = errorText.Error()
	w.WriteHeader(status)
	err := t.ExecuteTemplate(w, "error", t)
	if err != nil {
		log.Fatal("Error processing error handler:", err)
	}
}

// Execute given template and write it to the client
func (t *Templates) Exec(w http.ResponseWriter, name string, data interface{}) error {
	tmpl := t.Lookup(name)
	if tmpl == nil {
		err := fmt.Errorf("template %s not found", name)
		t.logError(w, http.StatusNotFound, err)
		return err
	}
	if err := tmpl.Execute(w, data); err != nil {
		err := fmt.Errorf("error processing template %s: %s", name, err)
		t.logError(w, http.StatusBadRequest, err)
		return err
	}
	return nil
}
