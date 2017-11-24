package tensorrt

import "runtime"

var (
	supportedSystem = runtime.GOOS == "linux" && (runtime.GOARCH == "amd64" || runtime.GOARCH == "arm64")
)
