package web

import (
	"errors"
	"fmt"
	"html/template"
	"image/color"
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
	Dropdown []Link
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
	t := &Templates{Menu: []Link{}, Options: []Link{}}
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

func (t *Templates) ToggleOption(name string) bool {
	for i, opt := range t.Options {
		if opt.Name == name {
			opt.Selected = !opt.Selected
			t.Options[i].Selected = opt.Selected
			return opt.Selected
		}
	}
	return false
}

// Return custom error response
func (t *Templates) ErrorHandler(status int, errorText error) http.Handler {
	return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		t.Toplevel = true
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
func (t *Templates) Exec(w http.ResponseWriter, name string, data interface{}, topLevel bool) error {
	t.Toplevel = topLevel
	tmpl := t.Lookup(name)
	if tmpl == nil {
		err := fmt.Errorf("template %s not found", name)
		t.logError(w, http.StatusNotFound, err)
		return err
	}
	if err := tmpl.Execute(w, data); err != nil {
		log.Printf("error processing template %s: %s", name, err)
	}
	return nil
}

// convert to #rrggbb color string
func htmlColor(col color.Color) string {
	r, g, b, _ := col.RGBA()
	return fmt.Sprintf("#%02x%02x%02x", r/0x101, g/0x101, b/0x101)
}
