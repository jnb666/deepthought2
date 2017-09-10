package nnet

import (
	"encoding/json"
	"fmt"
	"os"
	"path"
	"reflect"
	"strings"
)

// Training configuration settings
type Config struct {
	Eta           float64
	Lambda        float64
	NormalWeights bool
	Shuffle       bool
	FlattenInput  bool
	TrainBatch    int
	TestBatch     int
	MaxEpoch      int
	MaxSamples    int
	LogEvery      int
	MinLoss       float64
	RandSeed      int64
	Threads       int
	DebugLevel    int
	Layers        []LayerConfig
}

// Load network from json file under DataDir
func LoadConfig(name string) (c Config, err error) {
	filePath := path.Join(DataDir, name+".net")
	var f *os.File
	if f, err = os.Open(filePath); err != nil {
		return
	}
	defer f.Close()
	fmt.Println("loading network config from", name+".net")
	dec := json.NewDecoder(f)
	err = dec.Decode(&c)
	return
}

// Save default network definition to a json file under DataDir
func (c Config) SaveDefault(name string) error {
	err := c.Save(name + "_default")
	if err != nil {
		return err
	}
	if !FileExists(name + ".net") {
		err = c.Save(name)
	}
	return err
}

func (c Config) Save(name string) error {
	filePath := path.Join(DataDir, "."+name+".net")
	f, err := os.Create(filePath)
	if err != nil {
		return err
	}
	fmt.Println("saving network config to", name+".net")
	enc := json.NewEncoder(f)
	enc.SetIndent("", "  ")
	if err = enc.Encode(c); err != nil {
		f.Close()
		return err
	}
	f.Close()
	return os.Rename(filePath, path.Join(DataDir, name+".net"))
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

func (c Config) String() string {
	fields := c.Fields()
	str := make([]string, len(fields))
	for i, key := range fields {
		str[i] = fmt.Sprintf("%-14s: %v", key, c.Get(key))
	}
	return strings.Join(str, "\n")
}
