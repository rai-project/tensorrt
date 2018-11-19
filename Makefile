all: generate

fmt:
	go fmt ./...

install-deps:
	go get github.com/jteeuwen/go-bindata/...
	go get github.com/elazarl/go-bindata-assetfs/...
	go get github.com/golang/dep
	dep ensure -v

generate: clean generate-models

generate-models:
	go-bindata -nomemcopy -prefix builtin_models/ -pkg tensorrt -o builtin_models_static.go -ignore=.DS_Store  -ignore=README.md builtin_models/...

clean-models:
	rm -fr builtin_models_static.go

clean: clean-models

travis: install-deps glide-install generate
	echo "building..."
	go build
