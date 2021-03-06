package nnet

import (
	"encoding/json"
	"fmt"
	"log"
	"math"
	"os"
	"path"
	"reflect"
	"strconv"
	"strings"
)

var defaultConfig Config

type OptionList interface {
	Options() []string
	String() string
}

// Training configuration settings
type Config struct {
	DataSet      string
	Optimiser    OptimiserType
	Eta          float64
	EtaDecay     float64
	EtaDecayStep int
	Lambda       float64
	Momentum     float64
	WeightInit   InitType
	Bias         float64
	Shuffle      bool
	Normalise    bool
	Distort      bool
	TrainRuns    int
	TrainBatch   int
	TestBatch    int
	MaxEpoch     int
	MaxSeconds   int
	MaxSamples   int
	LogEvery     int
	StopAfter    int
	ExtraEpochs  int
	ValidEMA     float64
	MinLoss      float64
	RandSeed     int64
	DebugLevel   int
	UseGPU       bool
	Profile      bool
	MemProfile   bool
	Layers       []LayerConfig
}

// Load network from json file under DataDir
func LoadConfig(name string) (c Config, err error) {
	filePath := path.Join(DataDir, name)
	var f *os.File
	if f, err = os.Open(filePath); err != nil {
		return
	}
	defer f.Close()
	log.Println("loading network config from", name)
	dec := json.NewDecoder(f)
	err = dec.Decode(&c)
	return
}

func (c Config) DatasetConfig(test bool) DatasetOptions {
	opts := DatasetOptions{
		BatchSize:  c.TrainBatch,
		MaxSamples: c.MaxSamples,
		Normalise:  c.Normalise,
		Distort:    c.Distort,
	}
	if test {
		if c.TestBatch != 0 {
			opts.BatchSize = c.TestBatch
		}
		opts.Distort = false
	}
	return opts
}

// Get learning rate and weight decay
func (c Config) OptimiserParams(epoch, samples int) (learningRate, weightDecay float32) {
	if c.EtaDecay > 0 && c.EtaDecayStep > 0 {
		c.Eta *= math.Pow(c.EtaDecay, float64((epoch-1)/c.EtaDecayStep))
	}
	decay := c.Eta * c.Lambda / float64(samples)
	if epoch == 1 || (c.EtaDecay > 0 && c.EtaDecayStep > 0 && (epoch-1)%c.EtaDecayStep == 0) {
		log.Printf("learning rate=%.4g weight decay=%.4g\n", c.Eta, decay)
	}
	return float32(c.Eta), float32(decay)
}

func (c Config) Copy() Config {
	conf := c
	conf.Layers = append([]LayerConfig{}, c.Layers...)
	return conf
}

// Append layers to the config struct
func (c Config) AddLayers(layers ...ConfigLayer) Config {
	for _, l := range layers {
		c.Layers = append(c.Layers, l.Marshal())
	}
	return c
}

// Save config to JSON file under DataDir
func (c Config) Save(name string) error {
	filePath := path.Join(DataDir, name)
	f, err := os.Create(filePath)
	if err != nil {
		return err
	}
	defer f.Close()
	log.Println("saving network config to", name)
	enc := json.NewEncoder(f)
	enc.SetIndent("", "  ")
	return enc.Encode(c)
}

func (c Config) Fields() []string {
	st := reflect.TypeOf(c)
	fld := make([]string, st.NumField()-1)
	for i := range fld {
		fld[i] = st.Field(i).Name
	}
	return fld
}

func (c Config) Get(key string) interface{} {
	s := reflect.ValueOf(c)
	return s.FieldByName(key).Interface()
}

func (c Config) configString() string {
	fields := c.Fields()
	str := []string{"== Config =="}
	for _, key := range fields {
		val := c.Get(key)
		if !reflect.DeepEqual(val, defaultConfig.Get(key)) {
			str = append(str, fmt.Sprintf("%-14s: %v", key, val))
		}
	}
	return strings.Join(str, "\n")
}

func (c Config) String() string {
	s := c.configString()
	if c.Layers != nil {
		str := []string{"\n== Network =="}
		for i, layer := range c.Layers {
			l := layer.Unmarshal()
			str = append(str, fmt.Sprintf("%2d: %s", i, l))
			if group, ok := l.(LayerGroup); ok {
				desc := group.LayerDesc()
				for j, l := range group.Layers() {
					str = append(str, fmt.Sprintf("    %s %s", desc[j], l))
				}
			}
		}
		s += strings.Join(str, "\n")
	}
	return s
}

func (c Config) SetString(key, val string) (Config, error) {
	s := reflect.ValueOf(&c).Elem()
	f := s.FieldByName(key)
	var err error
	switch f.Type().Kind() {
	case reflect.Int, reflect.Int64:
		var x int64
		if x, err = strconv.ParseInt(val, 10, 64); err == nil {
			f.SetInt(x)
		}
	case reflect.Float64:
		var x float64
		if x, err = strconv.ParseFloat(val, 64); err == nil {
			f.SetFloat(x)
		}
	case reflect.String:
		f.SetString(val)
	case reflect.Bool:
		f.SetBool(val != "")
	default:
		return c, fmt.Errorf("invalid type for SetString: %v", f.Type().Kind())
	}
	return c, err
}
