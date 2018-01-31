package stats

import (
	"fmt"
	"html/template"
	"math"
)

// Calc exponentional moving average
type EMA float64

func (e EMA) Add(val, n float64) float64 {
	if e == 0 {
		return val
	}
	k := 2.0 / (n + 1.0)
	return val*k + float64(e)*(1-k)
}

// Running mean and stddev as per http://www.johndcook.com/blog/standard_deviation/
type Average struct {
	Count, Mean float64
	Var, StdDev float64
	oldM, oldV  float64
}

func (s *Average) Add(x float64) {
	s.Count++
	if s.Count == 1 {
		s.oldM, s.Mean = x, x
		s.oldV = 0
	} else {
		s.Mean = s.oldM + (x-s.oldM)/s.Count
		s.Var = s.oldV + (x-s.oldM)*(x-s.Mean)
		s.oldM, s.oldV = s.Mean, s.Var
		if s.Count > 1 {
			s.StdDev = math.Sqrt(s.Var / (s.Count - 1))
		}
	}
}

func (s *Average) HTML() template.HTML {
	var text string
	if s.Mean > 10 {
		if s.StdDev < 0.1 {
			text = fmt.Sprintf("%.1f", s.Mean)
		} else {
			text = fmt.Sprintf("%.1f&PlusMinus;%.1f", s.Mean, s.StdDev)
		}
	} else {
		if s.StdDev < 0.01 {
			text = fmt.Sprintf("%.2f", s.Mean)
		} else {
			text = fmt.Sprintf("%.2f&PlusMinus;%.2f", s.Mean, s.StdDev)
		}
	}
	return template.HTML(text)
}
