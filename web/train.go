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
	"gonum.org/v1/plot/vg/draw"
	"gonum.org/v1/plot/vg/vgsvg"
	"html/template"
	"log"
	"net/http"
	"sort"
	"strings"
)

var upgrader = websocket.Upgrader{
	ReadBufferSize:  1024,
	WriteBufferSize: 1024,
}

type TrainPage struct {
	*Templates
	net      *Network
	groups   map[string][]int
	keys     []string
	disabled map[string]bool
}

type HistoryRow struct {
	Id      int
	Params  template.HTML
	Runs    int
	Stats   []template.HTML
	Color   string
	Enabled bool
}

func init() {
	plotutil.DefaultColors = plotutil.DarkColors
}

// Base data for handler functions to perform network training and display the stats
func NewTrainPage(t *Templates, net *Network) *TrainPage {
	p := &TrainPage{net: net, Templates: t, disabled: map[string]bool{}}
	for _, opt := range []string{"loss", "errors", "history", "tune", "purge", "clear"} {
		p.AddOption(Link{Name: opt, Url: "/train/set/" + opt, Selected: opt == "errors"})
	}
	return p
}

// Handler function for the train base template
func (p *TrainPage) Base() func(w http.ResponseWriter, r *http.Request) {
	return func(w http.ResponseWriter, r *http.Request) {
		p.net.Lock()
		defer p.net.Unlock()
		p.Select("/train")
		p.Heading = p.net.heading()
		p.Exec(w, "train", p, true)
	}
}

// Handler function to toggle options
func (p *TrainPage) Setopt() func(w http.ResponseWriter, r *http.Request) {
	return func(w http.ResponseWriter, r *http.Request) {
		opt := mux.Vars(r)["opt"]
		p.net.Lock()
		defer p.net.Unlock()
		switch opt {
		default:
			p.ToggleOption(opt)
		case "tune":
			p.net.tuneMode = p.ToggleOption(opt)
		case "clear", "purge":
			if opt == "clear" {
				p.net.History = p.net.History[:0]
			} else {
				hNew := []HistoryData{}
				for _, h := range p.net.History {
					if !p.disabled[tuneParams(h)] {
						hNew = append(hNew, h)
					}
				}
				p.net.History = hNew
			}
			p.net.Export()
			if err := SaveNetwork(p.net.NetworkData, false); err != nil {
				p.logError(w, http.StatusBadRequest, err)
				return
			}
		}
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
				if err := p.net.Start(p.net.Conf, false); err != nil {
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
		p.getHistory()
		p.Exec(w, "stats", p, false)
	}
}

// Handler function for the form to filter the stats values
func (p *TrainPage) Filter() func(w http.ResponseWriter, r *http.Request) {
	return func(w http.ResponseWriter, r *http.Request) {
		p.net.Lock()
		defer p.net.Unlock()
		r.ParseForm()
		for i, key := range p.keys {
			p.disabled[key] = r.Form.Get(fmt.Sprintf("r%d", i)) == ""
		}
		http.Redirect(w, r, "/stats", http.StatusFound)
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

func (p *TrainPage) StatsHeaders() []string {
	return nnet.StatsHeaders(p.net.Data)
}

func (p *TrainPage) LatestStats(n int) []nnet.Stats {
	res := []nnet.Stats{}
	last := len(p.net.test.Stats) - 1
	for i := last; i >= 0 && i > last-n; i-- {
		res = append(res, p.net.test.Stats[i])
	}
	if len(p.net.test.Stats) < n && p.OptionSelected("history") && len(p.net.History) > 0 {
		last = len(p.net.History) - 1
		res = append(res, p.net.History[last].Stats)
	}
	return res
}

func (p *TrainPage) HistoryHeaders() []string {
	head := []string{"params", "epochs"}
	for _, h := range nnet.StatsHeaders(p.net.Data) {
		if strings.HasSuffix(h, " error") {
			head = append(head, h)
		}
	}
	return append(head, "run time")
}

// sort into groups for each set of runs with given params
func (p *TrainPage) getHistory() {
	p.keys = []string{}
	p.groups = make(map[string][]int)
	for i, h := range p.net.History {
		params := tuneParams(h)
		if val, ok := p.groups[params]; ok {
			p.groups[params] = append(val, i)
		} else {
			p.groups[params] = []int{i}
			p.keys = append(p.keys, params)
		}
	}
	sort.Strings(p.keys)
}

func (p *TrainPage) History() []HistoryRow {
	statsHead := p.StatsHeaders()
	table := []HistoryRow{}
	setId := 0
	for i, params := range p.keys {
		stats := new(statsTable)
		for _, i := range p.groups[params] {
			h := p.net.History[i]
			stats.add(float64(h.Stats.Epoch))
			for j, val := range h.Stats.Values {
				if strings.HasSuffix(statsHead[j], " error") {
					stats.add(100 * val)
				}
			}
			stats.add(h.Stats.Elapsed.Seconds())
			stats.next()
		}
		r := HistoryRow{
			Id:      i,
			Runs:    len(p.groups[params]),
			Params:  template.HTML(params),
			Enabled: !p.disabled[params],
		}
		if r.Enabled {
			r.Color = htmlColor(plotutil.Color(setId))
			setId++
		}
		for _, s := range stats.avg {
			r.Stats = append(r.Stats, s.HTML())
		}
		table = append(table, r)
	}
	return table
}

func (p *TrainPage) LossPlot(width, height int) template.HTML {
	plt := newPlot()
	plt.X.Label.Text = "epoch"
	plt.Y.Label.Text = "loss"
	line := newLinePlot(p.net.test.Stats, 0, 1)
	line.Color = plotutil.Color(0)
	plt.Add(line)
	plt.Legend.Add("training loss ", line)
	return writePlot(plt, width, height)
}

func (p *TrainPage) ErrorPlot(width, height int) template.HTML {
	plt := newPlot()
	plt.X.Label.Text = "epoch"
	plt.Y.Label.Text = "error %"
	for i, name := range p.StatsHeaders()[1:] {
		line := newLinePlot(p.net.test.Stats, i+1, 100)
		line.Color = plotutil.Color(i)
		plt.Add(line)
		plt.Legend.Add(name, line)
	}
	return writePlot(plt, width, height)
}

func (p *TrainPage) HistoryPlot(width, height int) template.HTML {
	plt := newPlot()
	plt.Legend.Left = true
	plt.X.Label.Text = "run time"
	ix := 0
	for i, name := range p.StatsHeaders() {
		if strings.HasSuffix(name, " error") {
			ix = i
			plt.Y.Label.Text = name + " %"
		}
	}
	var pt struct{ X, Y float64 }
	setId := 0
	for _, params := range p.keys {
		if p.disabled[params] {
			continue
		}
		var pts plotter.XYs
		for _, i := range p.groups[params] {
			h := p.net.History[i]
			if ix < len(h.Stats.Values) {
				pt.X = h.Stats.Elapsed.Seconds()
				pt.Y = 100 * h.Stats.Values[ix]
				pts = append(pts, pt)
			}
		}
		s, _ := plotter.NewScatter(pts)
		s.GlyphStyle.Shape = draw.BoxGlyph{}
		s.GlyphStyle.Color = plotutil.Color(setId)
		s.GlyphStyle.Radius *= 1.5
		plt.Add(s)
		setId++
	}
	return writePlot(plt, width, height)
}

func newPlot() *plot.Plot {
	p, err := plot.New()
	if err != nil {
		log.Fatal("Plot error: ", err)
	}
	fontSmall := newFont(10)
	p.X.Padding, p.Y.Padding = 0, 0
	p.X.Label.Font = fontSmall
	p.Y.Label.Font = fontSmall
	p.X.Tick.Label.Font = fontSmall
	p.Y.Tick.Label.Font = fontSmall
	p.Legend.Top = true
	p.Legend.Font = fontSmall
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

type statsTable struct {
	avg []*nnet.Average
	col int
	row int
}

func (s *statsTable) add(x float64) {
	if s.row == 0 {
		s.avg = append(s.avg, new(nnet.Average))
	}
	s.avg[s.col].Add(x)
	s.col++
}

func (s *statsTable) next() {
	s.row++
	s.col = 0
}
