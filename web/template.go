package web

import (
	"fmt"
	"github.com/gorilla/sessions"
	"html/template"
	"log"
	"net/http"
	"os"
	"strings"
)

var AssetDir = os.Getenv("GOPATH") + "/src/github.com/jnb666/deepthought2/assets"

var authKey = []byte("uekahcahziem2Tha")

// Template and main menu definition
type Templates struct {
	*template.Template
	Menu    []Link
	Options []Link
	store   sessions.Store
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
	if err != nil {
		return nil, err
	}
	t.store = sessions.NewCookieStore(authKey)
	return t, err
}

func (t *Templates) Clone() *Templates {
	return &Templates{
		Template: t.Template,
		Menu:     append([]Link{}, t.Menu...),
		Options:  append([]Link{}, t.Options...),
		store:    t.store,
	}
}

func (t *Templates) Select(url string) *Templates {
	for i, key := range t.Menu {
		t.Menu[i].Selected = strings.HasPrefix(key.Url, url)
	}
	return t
}

func (t *Templates) AddMenuItem(l Link) *Templates {
	t.Menu = append(t.Menu, l)
	return t
}

func (t *Templates) AddOption(l Link) *Templates {
	t.Options = append(t.Options, l)
	return t
}

func (t *Templates) SelectOptions(names []string) *Templates {
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
