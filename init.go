package tensorrt

import (
	"github.com/rai-project/config"
	"github.com/rai-project/logger"
	"github.com/sirupsen/logrus"
)

var (
	log *logrus.Entry
)

func init() {
	config.AfterInit(func() {
		log = logger.New().WithField("pkg", "tensorrt")
		if !supportedSystem {
			log.Error("tensorrt is only available on linux/amd64 and linux/arm64. not registering tensorrt")
		}
	})
}
