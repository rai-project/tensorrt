package predict

import (
	"bufio"
	"os"
	"strings"

	"github.com/k0kubun/pp"
	opentracing "github.com/opentracing/opentracing-go"
	olog "github.com/opentracing/opentracing-go/log"
	"github.com/pkg/errors"
	"github.com/rai-project/config"
	"github.com/rai-project/dlframework"
	"github.com/rai-project/dlframework/framework/agent"
	"github.com/rai-project/dlframework/framework/options"
	common "github.com/rai-project/dlframework/framework/predict"
	"github.com/rai-project/downloadmanager"
	gotensorrt "github.com/rai-project/go-tensorrt"
	"github.com/rai-project/image"
	"github.com/rai-project/image/types"
	"github.com/rai-project/tensorrt"
	"github.com/rai-project/tracer"
	"github.com/rai-project/tracer/ctimer"
	context "golang.org/x/net/context"
)

// ImagePredictor ...
type ImagePredictor struct {
	common.ImagePredictor
	features  []string
	predictor *gotensorrt.Predictor
	inputDims []uint32
}

// New ...
func New(model dlframework.ModelManifest, opts ...options.Option) (common.Predictor, error) {
	modelInputs := model.GetInputs()
	if len(modelInputs) != 1 {
		return nil, errors.New("number of inputs not supported")
	}

	firstInputType := modelInputs[0].GetType()
	if strings.ToLower(firstInputType) != "image" {
		return nil, errors.New("input type not supported")
	}

	predictor := new(ImagePredictor)

	return predictor.Load(context.Background(), model, opts...)
}

// Load ...
func (p *ImagePredictor) Load(ctx context.Context, model dlframework.ModelManifest, opts ...options.Option) (common.Predictor, error) {
	span, ctx := tracer.StartSpanFromContext(ctx, tracer.STEP_TRACE, "Load")
	defer span.Finish()

	framework, err := model.ResolveFramework()
	if err != nil {
		return nil, err
	}

	workDir, err := model.WorkDir()
	if err != nil {
		return nil, err
	}

	ip := &ImagePredictor{
		ImagePredictor: common.ImagePredictor{
			Base: common.Base{
				Framework: framework,
				Model:     model,
				Options:   options.New(opts...),
			},
			WorkDir: workDir,
		},
	}

	if err = ip.download(ctx); err != nil {
		return nil, err
	}

	if err = ip.loadPredictor(ctx); err != nil {
		return nil, err
	}

	return ip, nil
}

// GetPreprocessOptions ...
func (p *ImagePredictor) GetPreprocessOptions(ctx context.Context) (common.PreprocessOptions, error) {
	mean, err := p.GetMeanImage()
	if err != nil {
		return common.PreprocessOptions{}, err
	}

	scale, err := p.GetScale()
	if err != nil {
		return common.PreprocessOptions{}, err
	}

	imageDims, err := p.GetImageDimensions()
	if err != nil {
		return common.PreprocessOptions{}, err
	}

	return common.PreprocessOptions{
		Context:   ctx,
		MeanImage: mean,
		Scale:     scale,
		Size:      []int{int(imageDims[1]), int(imageDims[2])},
		ColorMode: types.BGRMode,
		Layout:    image.CHWLayout,
	}, nil
}

func (p *ImagePredictor) download(ctx context.Context) error {
	span, ctx := tracer.StartSpanFromContext(ctx,
		tracer.STEP_TRACE,
		"Download",
		opentracing.Tags{
			"graph_url":           p.GetGraphUrl(),
			"target_graph_file":   p.GetGraphPath(),
			"weights_url":         p.GetWeightsUrl(),
			"target_weights_file": p.GetWeightsPath(),
			"feature_url":         p.GetFeaturesUrl(),
			"target_feature_file": p.GetFeaturesPath(),
		},
	)
	defer span.Finish()

	model := p.Model
	if model.Model.IsArchive {
		baseURL := model.Model.BaseUrl
		span.LogFields(
			olog.String("event", "download model archive"),
		)
		_, err := downloadmanager.DownloadInto(baseURL, p.WorkDir, downloadmanager.Context(ctx))
		if err != nil {
			return errors.Wrapf(err, "failed to download model archive from %v", model.Model.BaseUrl)
		}
		return nil
	}
	checksum := p.GetGraphChecksum()
	if checksum == "" {
		return errors.New("Need graph file checksum in the model manifest")
	}

	span.LogFields(
		olog.String("event", "download graph"),
	)
	if _, err := downloadmanager.DownloadFile(p.GetGraphUrl(), p.GetGraphPath(), downloadmanager.MD5Sum(checksum)); err != nil {
		return err
	}

	checksum = p.GetWeightsChecksum()
	if checksum == "" {
		return errors.New("Need weights file checksum in the model manifest")
	}

	span.LogFields(
		olog.String("event", "download weights"),
	)
	if _, err := downloadmanager.DownloadFile(p.GetWeightsUrl(), p.GetWeightsPath(), downloadmanager.MD5Sum(checksum)); err != nil {
		return err
	}

	checksum = p.GetFeaturesChecksum()
	if checksum == "" {
		return errors.New("Need features file checksum in the model manifest")
	}

	span.LogFields(
		olog.String("event", "download features"),
	)
	if _, err := downloadmanager.DownloadFile(p.GetFeaturesUrl(), p.GetFeaturesPath(), downloadmanager.MD5Sum(checksum)); err != nil {
		return err
	}

	return nil
}

func (p *ImagePredictor) loadPredictor(ctx context.Context) error {
	span, ctx := tracer.StartSpanFromContext(ctx, tracer.STEP_TRACE, "LoadPredictor")

	defer span.Finish()

	span.LogFields(
		olog.String("event", "read features"),
	)

	var features []string
	f, err := os.Open(p.GetFeaturesPath())
	if err != nil {
		return errors.Wrapf(err, "cannot read %s", p.GetFeaturesPath())
	}
	defer f.Close()
	scanner := bufio.NewScanner(f)
	for scanner.Scan() {
		line := scanner.Text()
		features = append(features, line)
	}
	p.features = features

	p.inputDims, err = p.GetImageDimensions()
	if err != nil {
		return err
	}

	span.LogFields(
		olog.String("event", "creating predictor"),
	)

	opts, err := p.GetPredictionOptions(ctx)
	if err != nil {
		return err
	}

	pred, err := gotensorrt.New(
		options.WithOptions(opts),
		options.Graph([]byte(p.GetGraphPath())),
		options.Weights([]byte(p.GetWeightsPath())),
	)
	if err != nil {
		return err
	}
	p.predictor = pred

	return nil
}

// Predict ...
func (p *ImagePredictor) Predict(ctx context.Context, data [][]float32, opts ...options.Option) ([]dlframework.Features, error) {
	if p.TraceLevel() >= tracer.FRAMEWORK_TRACE {
		err := p.predictor.StartProfiling("tensorrt", "predict")
		if err != nil {
			log.WithError(err).WithField("framework", "tensorrt").Error("unable to start framework profiling")
		} else {
			defer func() {
				p.predictor.EndProfiling()
				profBuffer, err := p.predictor.ReadProfile()
				if err != nil {
					pp.Println(err)
					return
				}

				t, err := ctimer.New(profBuffer)
				if err != nil {
					pp.Println(err)
					return
				}
				t.Publish(ctx)

				p.predictor.DisableProfiling()
			}()
		}
	}

	var input []float32
	for _, v := range data {
		input = append(input, v...)
	}

	predictions, err := p.predictor.Predict(input, int(p.BatchSize()), int(p.inputDims[0]), int(p.inputDims[1]), int(p.inputDims[2]))
	if err != nil {
		return nil, err
	}

	var output []dlframework.Features
	batchSize := int(p.BatchSize())
	length := len(predictions) / batchSize

	for i := 0; i < batchSize; i++ {
		rprobs := make([]*dlframework.Feature, length)
		for j := 0; j < length; j++ {
			rprobs[j] = &dlframework.Feature{
				Index:       int64(j),
				Name:        p.features[j],
				Probability: predictions[i*length+j].Probability,
			}
		}
		output = append(output, rprobs)
	}
	return output, nil
}

// Reset ...
func (p *ImagePredictor) Reset(ctx context.Context) error {

	return nil
}

// Close ...
func (p *ImagePredictor) Close() error {
	if p.predictor != nil {
		p.predictor.Close()
	}
	return nil
}

func init() {
	config.AfterInit(func() {
		framework := tensorrt.FrameworkManifest
		agent.AddPredictor(framework, &ImagePredictor{
			ImagePredictor: common.ImagePredictor{
				Base: common.Base{
					Framework: framework,
				},
			},
		})
	})
}
