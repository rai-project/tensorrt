package main

import (
	"fmt"
	"os"

	"github.com/rai-project/config"
	cmd "github.com/rai-project/dlframework/framework/cmd/server"
	"github.com/rai-project/logger"
	"github.com/rai-project/tensorrt"
	_ "github.com/rai-project/tensorrt/predictor"
	"github.com/rai-project/tracer"
	"github.com/sirupsen/logrus"
)

var (
	modelName    string
	modelVersion string
	hostName, _  = os.Hostname()
	framework    = tensorrt.FrameworkManifest
	log          *logrus.Entry
)

func main() {
	rootCmd, err := cmd.NewRootCommand(tensorrt.Register, framework)
	if err != nil {
		fmt.Println(err)
		os.Exit(-1)
	}

	defer tracer.Close()
	if err := rootCmd.Execute(); err != nil {
		fmt.Println(err)
		os.Exit(-1)
	}
}

func init() {
	config.AfterInit(func() {
		log = logger.New().WithField("pkg", "tensorrt-agent")
	})
}
