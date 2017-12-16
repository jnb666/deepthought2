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
	"time"
)

var upgrader = websocket.Upgrader{
	ReadBufferSize:  1024,
	WriteBufferSize: 1024,
}

type TrainPage struct {
	*Templates
	net *Network
}

// Base data for handler functions to perform network training and display the stats
func NewTrainPage(t *Templates, net *Network) *TrainPage {
	p := &TrainPage{net: net}
	p.Templates = t.Select("/train")
	p.AddOption(Link{Name: "start", Url: "/train/start"})
	p.AddOption(Link{Name: "stop", Url: "/train/stop"})
	p.AddOption(Link{Name: "continue", Url: "/train/continue"})
	return p
}

// Handler function for the train template
func (p *TrainPage) Base() func(w http.ResponseWriter, r *http.Request) {
	return func(w http.ResponseWriter, r *http.Request) {
		cmd := mux.Vars(r)["cmd"]
		//log.Printf("trainBase: %s cmd=%s", r.URL.Path, cmd)
		p.net.Lock()
		defer p.net.Unlock()
		switch cmd {
		case "start", "continue":
			if p.net.running {
				log.Println("skip start - already running")
			} else {
				p.net.Train(cmd == "start")
			}
			http.Redirect(w, r, "/train/stats", http.StatusFound)
		case "stop":
			p.net.running = false
			http.Redirect(w, r, "/train/stats", http.StatusFound)
		default:
			if err := p.ExecuteTemplate(w, "train", p); err != nil {
				logError(w, err)
			}
		}
	}
}

// Handler function for the stats frame
func (p *TrainPage) Stats() func(w http.ResponseWriter, r *http.Request) {
	return func(w http.ResponseWriter, r *http.Request) {
		p.net.Lock()
		defer p.net.Unlock()
		//log.Println("trainStats:", r.URL.Path)
		if err := p.ExecuteTemplate(w, "stats", p); err != nil {
			logError(w, err)
		}
	}
}

// Handler function for websocket connection
func (p *TrainPage) Websocket() func(w http.ResponseWriter, r *http.Request) {
	return func(w http.ResponseWriter, r *http.Request) {
		var err error
		p.net.conn, err = upgrader.Upgrade(w, r, nil)
		if err != nil {
			logError(w, err)
		}
	}
}

func (p *TrainPage) Heading() template.HTML {
	s := fmt.Sprintf(`%s: epoch <span id="epoch">%d</span> of %d`, p.net.Conf.Model, p.net.Epoch, p.net.MaxEpoch)
	return template.HTML(s)
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
	plt := newPlot()
	line := newLinePlot(p.net.base.Stats, 0, 1)
	plt.Add(line)
	plt.Legend.Add("training loss ", line)
	return writePlot(plt, width, height)
}

func (p *TrainPage) ErrorPlot(width, height int) template.HTML {
	plt := newPlot()
	for i, name := range p.Headers()[1:] {
		line := newLinePlot(p.net.base.Stats, i+1, 100)
		plt.Add(line)
		plt.Legend.Add(name+" % ", line)
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
