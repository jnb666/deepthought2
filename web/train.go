package web

import (
	"bytes"
	"fmt"
	"github.com/gorilla/mux"
	"github.com/gorilla/websocket"
	"github.com/jnb666/deepthought2/nnet"
	"github.com/jnb666/deepthought2/stats"
	"gonum.org/v1/plot"
	"gonum.org/v1/plot/palette"
	"gonum.org/v1/plot/palette/moreland"
	"gonum.org/v1/plot/plotter"
	"gonum.org/v1/plot/plotutil"
	"gonum.org/v1/plot/vg"
	"gonum.org/v1/plot/vg/draw"
	"gonum.org/v1/plot/vg/vgsvg"
	"html/template"
	"image/color"
	"log"
	"net/http"
	"sort"
	"strings"
)

const (
	plotWidth  = 450
	plotHeight = 300
)

var upgrader = websocket.Upgrader{
	ReadBufferSize:  1024,
	WriteBufferSize: 1024,
}

type TrainPage struct {
	*Templates
	Plots      []template.HTML
	PlotWidth  int
	PlotHeight int
	net        *Network
	groups     map[string][]int
	keys       []string
	disabled   map[string]bool
}

type HistoryRow struct {
	Id      int
	Params  template.HTML
	Runs    int
	Stats   []template.HTML
	Color   string
	Enabled bool
}

// Base data for handler functions to perform network training and display the stats
func NewTrainPage(t *Templates, net *Network) *TrainPage {
	p := &TrainPage{net: net, Templates: t, disabled: map[string]bool{}, PlotWidth: plotWidth + 32, PlotHeight: plotHeight}
	for _, opt := range []string{"loss", "errors", "matrix", "history", "tune", "purge", "clear"} {
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
		p.Plots = []template.HTML{}
		if p.OptionSelected("loss") {
			p.Plots = append(p.Plots, p.lossPlot())
		}
		if p.OptionSelected("errors") {
			p.Plots = append(p.Plots, p.errorPlot())
		}
		if p.OptionSelected("matrix") {
			p.Plots = append(p.Plots, p.matrixPlot())
		}
		if p.OptionSelected("history") {
			p.Plots = append(p.Plots, p.historyPlot())
		}
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
	logEvery := p.net.LogEvery
	if logEvery == 0 {
		logEvery = 1
	}
	res := []nnet.Stats{}
	last := len(p.net.test.Stats) - 1
	if last >= 0 {
		res = append(res, p.net.test.Stats[last])
		for i := logEvery*(last/logEvery) - 1; i >= 0 && len(res) < n; i -= logEvery {
			res = append(res, p.net.test.Stats[i])
		}
	}
	if len(res) < n && p.OptionSelected("history") && len(p.net.History) > 0 {
		res = append(res, p.net.History[len(p.net.History)-1].Stats)
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

func (p *TrainPage) lossPlot() template.HTML {
	plt := newPlot()
	plt.Add(plotter.NewGrid())
	plt.X.Label.Text = "epoch"
	plt.Y.Label.Text = "loss"
	line := newLinePlot(p.net.test.Stats, 0, 1)
	line.Color = plotutil.Color(0)
	plt.Add(line)
	plt.Legend.Add("training loss ", line)
	return writePlot(plt)
}

func (p *TrainPage) errorPlot() template.HTML {
	plt := newPlot()
	plt.Add(plotter.NewGrid())
	plt.X.Label.Text = "epoch"
	plt.Y.Label.Text = "error %"
	for i, name := range p.StatsHeaders()[1:] {
		line := newLinePlot(p.net.test.Stats, i+1, 100)
		line.Color = plotutil.Color(i)
		plt.Add(line)
		plt.Legend.Add(name, line)
	}
	return writePlot(plt)
}

func (p *TrainPage) matrixPlot() template.HTML {
	dset := "test"
	data, ok := p.net.Data[dset]
	if !ok {
		dset = "train"
		data = p.net.Data[dset]
	}
	nclass := len(data.Classes())
	hmap := newHeatMap(nclass, p.net.Labels[dset], p.net.Pred[dset], moreland.BlackBody())
	plt := newPlot()
	plt.X.Tick.Length = 0
	plt.Y.Tick.Length = 0
	plt.X.Tick.Marker = gridTicks{classes: data.Classes()}
	plt.Y.Tick.Marker = gridTicks{classes: data.Classes(), reverse: true}
	plt.X.Label.Text = "predicted"
	plt.Y.Label.Text = "class"
	plt.Add(hmap)
	style := draw.TextStyle{Font: newFont(10), Color: color.Gray{0xc0}}
	plt.Add(hmap.labels(style, 0.5))
	return writePlot(plt)
}

func (p *TrainPage) historyPlot() template.HTML {
	plt := newPlot()
	plt.Add(plotter.NewGrid())
	plt.Legend.Left = true
	plt.X.Label.Text = "run time"
	ix := 1
	head := p.StatsHeaders()
	if len(head) >= 3 {
		ix = 2
	}
	plt.Y.Label.Text = head[ix] + " %"
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
	return writePlot(plt)
}

func newPlot() *plot.Plot {
	p, err := plot.New()
	if err != nil {
		log.Fatal("Plot error: ", err)
	}
	bgColor := color.Gray{Y: 0x20}
	fontSmall := newFont(10)
	for _, ax := range []*plot.Axis{&p.X, &p.Y} {
		ax.Color = color.White
		ax.Label.Color = color.White
		ax.Tick.Color = color.White
		ax.Tick.Label.Color = color.White
		ax.Padding = 0
		ax.Label.Font = fontSmall
		ax.Tick.Label.Font = fontSmall
	}
	p.BackgroundColor = bgColor
	p.Legend.Top = true
	p.Legend.Font = fontSmall
	p.Legend.Color = color.White
	return p
}

func writePlot(p *plot.Plot) template.HTML {
	var buf bytes.Buffer
	writer, err := p.WriterTo(vg.Inch*vg.Length(plotWidth)/vgsvg.DPI, vg.Inch*vg.Length(plotHeight)/vgsvg.DPI, "svg")
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

type xy struct{ X, Y float64 }

func newLinePlot(stats []nnet.Stats, ix int, scale float64) linePlot {
	var pts plotter.XYs
	xmax, ymax := 1.0, 0.0
	for _, s := range stats {
		pt := xy{float64(s.Epoch), s.Values[ix] * scale}
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
	avg []*stats.Average
	col int
	row int
}

func (s *statsTable) add(x float64) {
	if s.row == 0 {
		s.avg = append(s.avg, new(stats.Average))
	}
	s.avg[s.col].Add(x)
	s.col++
}

func (s *statsTable) next() {
	s.row++
	s.col = 0
}

type heatMap struct {
	*plotter.HeatMap
	xmin, xmax, ymin, ymax float64
}

func newHeatMap(nclass int, labels, predict []int32, cmap palette.ColorMap) heatMap {
	g := grid{
		rows: nclass, cols: nclass,
		vals:  make([]int, nclass*nclass),
		total: make([]int, nclass),
	}
	for i, label := range labels {
		g.vals[nclass*int(label)+int(predict[i])]++
		g.total[label]++
	}
	cmap.SetMin(0)
	cmap.SetMax(1)
	p := cmap.Palette(256)
	return heatMap{
		HeatMap: &plotter.HeatMap{
			GridXYZ:   g,
			Palette:   p,
			Underflow: p.Colors()[0],
			Overflow:  p.Colors()[255],
			Min:       0,
			Max:       1,
		},
		xmin: g.X(0) - 0.5, xmax: g.X(g.cols-1) + 1.5,
		ymin: g.Y(0) - 0.5, ymax: g.Y(g.rows-1) + 0.5,
	}
}

func (h heatMap) DataRange() (xmin, xmax, ymin, ymax float64) {
	return h.xmin, h.xmax, h.ymin, h.ymax
}

func (h heatMap) labels(style draw.TextStyle, threshold float64) *plotter.Labels {
	lab := new(plotter.Labels)
	g := h.GridXYZ.(grid)
	for r := 0; r < g.rows; r++ {
		for c := 0; c < g.cols; c++ {
			val := g.vals[(g.rows-r-1)*g.cols+c]
			lab.Labels = append(lab.Labels, fmt.Sprint(val))
			lab.XYs = append(lab.XYs, xy{g.X(c) - 0.35, g.Y(r) - 0.15})
			s := style
			if g.Z(c, r) > threshold {
				s.Color = color.Black
			}
			lab.TextStyle = append(lab.TextStyle, s)
		}
		errorRow := 1 - g.Z(g.rows-r-1, r)
		lab.XYs = append(lab.XYs, xy{g.X(g.cols) - 0.4, g.Y(r) - 0.2})
		lab.Labels = append(lab.Labels, fmt.Sprintf("%.2g%%", 100*errorRow))
		lab.TextStyle = append(lab.TextStyle, style)
	}
	return lab
}

type grid struct {
	cols  int
	rows  int
	vals  []int
	total []int
}

func (g grid) Dims() (c, r int) { return g.cols, g.rows }

func (g grid) X(c int) float64 { return float64(c) }

func (g grid) Y(r int) float64 { return float64(r) }

func (g grid) Z(c, r int) float64 {
	row := g.rows - r - 1
	return float64(g.vals[row*g.cols+c]) / float64(g.total[row])
}

type gridTicks struct {
	classes []string
	reverse bool
}

func (t gridTicks) Ticks(min, max float64) []plot.Tick {
	ticks := make([]plot.Tick, len(t.classes))
	for i := range ticks {
		if t.reverse {
			ticks[i] = plot.Tick{Value: float64(i), Label: t.classes[len(t.classes)-i-1]}
		} else {
			ticks[i] = plot.Tick{Value: float64(i), Label: t.classes[i]}
		}
	}
	return ticks
}
