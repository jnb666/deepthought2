package web

import (
	"bytes"
	"fmt"
	"github.com/gorilla/mux"
	"github.com/gorilla/websocket"
	"github.com/jnb666/deepthought2/nnet"
	"gonum.org/v1/plot"
	"gonum.org/v1/plot/plotter"
	"gonum.org/v1/plot/plotutil"
	"gonum.org/v1/plot/vg"
	"gonum.org/v1/plot/vg/vgsvg"
	"html/template"
	"log"
	"net/http"
	"strconv"
	"time"
)

var upgrader = websocket.Upgrader{
	ReadBufferSize:  1024,
	WriteBufferSize: 1024,
}

type TrainPage struct {
	*Templates
	net     *Network
	disable map[int]bool
}

// Base data for handler functions to perform network training and display the stats
func NewTrainPage(t *Templates, net *Network) *TrainPage {
	p := &TrainPage{net: net, Templates: t, disable: map[int]bool{0: true}}
	return p
}

// Handler function for the train base template
func (p *TrainPage) Base() func(w http.ResponseWriter, r *http.Request) {
	return func(w http.ResponseWriter, r *http.Request) {
		p.net.Lock()
		defer p.net.Unlock()
		p.Options = []Link{}
		for i, opt := range p.Headers() {
			p.AddOption(Link{Name: opt, Url: fmt.Sprintf("/train/set/%d", i), Selected: !p.disable[i]})
		}
		p.Toplevel = true
		p.Select("/train")
		p.Heading = p.net.heading()
		p.Exec(w, "train", p)
	}
}

// Handler function to toggle options
func (p *TrainPage) Setopt() func(w http.ResponseWriter, r *http.Request) {
	return func(w http.ResponseWriter, r *http.Request) {
		id, _ := strconv.Atoi(mux.Vars(r)["id"])
		p.net.Lock()
		defer p.net.Unlock()
		p.disable[id] = !p.disable[id]
		http.Redirect(w, r, "/train", http.StatusFound)
	}
}

// Handler function for the train command options
func (p *TrainPage) Command() func(w http.ResponseWriter, r *http.Request) {
	return func(w http.ResponseWriter, r *http.Request) {
		cmd := mux.Vars(r)["cmd"]
		p.net.Lock()
		defer p.net.Unlock()
		switch cmd {
		case "start", "continue":
			if p.net.running {
				log.Println("skip start - already running")
			} else {
				err := p.net.Train(cmd == "start")
				if err != nil {
					p.logError(w, http.StatusInternalServerError, err)
					return
				}
			}
		case "stop":
			p.net.stop = true
		case "reset":
			if p.net.running {
				log.Println("skip reset - trainer is running")
			} else {
				if err := p.net.Start(); err != nil {
					p.logError(w, http.StatusInternalServerError, err)
					return
				}
			}
		}
		next := "/train"
		for _, item := range p.Menu {
			if item.Selected {
				next = item.Url
			}
		}

		http.Redirect(w, r, next, http.StatusFound)
	}
}

// Handler function for the stats frame
func (p *TrainPage) Stats() func(w http.ResponseWriter, r *http.Request) {
	return func(w http.ResponseWriter, r *http.Request) {
		p.net.Lock()
		defer p.net.Unlock()
		p.Toplevel = false
		p.Exec(w, "stats", p)
	}
}

// Handler function for websocket connection
func (p *TrainPage) Websocket() func(w http.ResponseWriter, r *http.Request) {
	return func(w http.ResponseWriter, r *http.Request) {
		var err error
		p.net.conn, err = upgrader.Upgrade(w, r, nil)
		if err != nil {
			log.Println("websocket connection error:", err)
		}
	}
}

func (p *TrainPage) Headers() []string {
	return nnet.StatsHeaders(p.net.Data)
}

func (p *TrainPage) LatestStats(n int) []nnet.Stats {
	last := len(p.net.base.Stats) - 1
	res := []nnet.Stats{}
	for i := last; i >= 0 && i > last-n; i-- {
		res = append(res, p.net.base.Stats[i])
	}
	return res
}

func (p *TrainPage) RunTime() string {
	if len(p.net.base.Stats) == 0 {
		return ""
	}
	elapsed := p.net.base.Stats[len(p.net.base.Stats)-1].Elapsed
	return fmt.Sprintf("run time: %s", elapsed.Round(10*time.Millisecond))
}

func (p *TrainPage) LossPlot(width, height int) template.HTML {
	if p.disable[0] {
		return ""
	}
	plt := newPlot()
	line := newLinePlot(p.net.base.Stats, 0, 1)
	plt.Add(line)
	plt.Legend.Add("training loss ", line)
	return writePlot(plt, width, height)
}

func (p *TrainPage) ErrorPlot(width, height int) template.HTML {
	lines := 0
	plt := newPlot()
	for i, name := range p.Headers()[1:] {
		if p.disable[i+1] {
			continue
		}
		line := newLinePlot(p.net.base.Stats, i+1, 100)
		plt.Add(line)
		plt.Legend.Add(name+" % ", line)
		lines++
	}
	if lines == 0 {
		return ""
	}
	return writePlot(plt, width, height)
}

func newPlot() *plot.Plot {
	p, err := plot.New()
	if err != nil {
		log.Fatal("Plot error: ", err)
	}
	fontSmall := newFont(10)
	fontMedium := newFont(12)
	p.X.Padding, p.Y.Padding = 0, 0
	p.X.Tick.Label.Font = fontSmall
	p.Y.Tick.Label.Font = fontSmall
	p.Legend.Top = true
	p.Legend.Font = fontMedium
	p.Add(plotter.NewGrid())
	return p
}

func writePlot(p *plot.Plot, w, h int) template.HTML {
	var buf bytes.Buffer
	writer, err := p.WriterTo(vg.Inch*vg.Length(w)/vgsvg.DPI, vg.Inch*vg.Length(h)/vgsvg.DPI, "svg")
	if err != nil {
		log.Fatal("Error writing plot: ", err)
	}
	writer.WriteTo(&buf)
	return template.HTML(buf.String())
}

func newFont(size vg.Length) vg.Font {
	font, err := vg.MakeFont("Helvetica", size)
	if err != nil {
		log.Fatal("Plot: failed loading font", err)
	}
	return font
}

func newLinePlot(stats []nnet.Stats, ix int, scale float64) linePlot {
	var pt struct{ X, Y float64 }
	var pts plotter.XYs
	xmax, ymax := 1.0, 0.0
	for _, s := range stats {
		pt.X, pt.Y = float64(s.Epoch), s.Values[ix]*scale
		pts = append(pts, pt)
		if pt.X > xmax {
			xmax = pt.X
		}
		if pt.Y > ymax {
			ymax = pt.Y
		}
	}
	l, _ := plotter.NewLine(pts)
	l.Width = 2
	l.Color = plotutil.Color(ix)
	return linePlot{Line: l, xmin: 1, xmax: xmax, ymin: 0, ymax: ymax}
}

// modified plotter.Line with a fixed scale
type linePlot struct {
	*plotter.Line
	xmin, xmax, ymin, ymax float64
}

func (l linePlot) DataRange() (xmin, xmax, ymin, ymax float64) {
	return l.xmin, l.xmax, l.ymin, l.ymax
}
