package nnet

import (
	"encoding/json"
	"fmt"
	"os"
	"path"
	"reflect"
	"strconv"
	"strings"
)

// Training configuration settings
type Config struct {
	DataSet       string
	Eta           float64
	Lambda        float64
	Bias          float64
	NormalWeights bool
	FlattenInput  bool
	Shuffle       bool
	TrainBatch    int
	TestBatch     int
	MaxEpoch      int
	MaxSamples    int
	LogEvery      int
	StopAfter     int
	MinLoss       float64
	RandSeed      int64
	DebugLevel    int
	UseGPU        bool
	Profile       bool
	Layers        []LayerConfig
}

// Load network from json file under DataDir
func LoadConfig(name string) (c Config, err error) {
	filePath := path.Join(DataDir, name)
	var f *os.File
	if f, err = os.Open(filePath); err != nil {
		return
	}
	defer f.Close()
	fmt.Println("loading network config from", name)
	dec := json.NewDecoder(f)
	err = dec.Decode(&c)
	return
}

// Append layers to the config struct
func (c Config) AddLayers(layers ...ConfigLayer) Config {
	for _, l := range layers {
		c.Layers = append(c.Layers, l.Marshal())
	}
	return c
}

// Save default network definition and overwites current config
func (c Config) SaveDefault(name string) error {
	err := c.Save(name + ".default")
	if err != nil {
		return err
	}
	err = c.Save(name + ".net")
	return err
}

// Save config to JSON file under DataDir
func (c Config) Save(name string) error {
	filePath := path.Join(DataDir, "."+name)
	f, err := os.Create(filePath)
	if err != nil {
		return err
	}
	fmt.Println("saving network config to", name)
	enc := json.NewEncoder(f)
	enc.SetIndent("", "  ")
	if err = enc.Encode(c); err != nil {
		f.Close()
		return err
	}
	f.Close()
	return os.Rename(filePath, path.Join(DataDir, name))
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
		str = append(str, fmt.Sprintf("%-14s: %v", key, c.Get(key)))
	}
	return strings.Join(str, "\n")
}

func (c Config) String() string {
	s := c.configString()
	if c.Layers != nil {
		str := []string{"\n== Network =="}
		for i, layer := range c.Layers {
			str = append(str, fmt.Sprintf("%2d: %s", i, layer))
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
	default:
		return c, fmt.Errorf("invalid type for SetString: %v", f.Type().Kind())
	}
	return c, err
}

func (c Config) SetBool(key string, val bool) (Config, error) {
	s := reflect.ValueOf(&c).Elem()
	f := s.FieldByName(key)
	if f.Type().Kind() == reflect.Bool {
		f.SetBool(val)
		return c, nil
	}
	return c, fmt.Errorf("invalid type for SetBool: %v", f.Type().Kind())
}
