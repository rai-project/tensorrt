package predict

import (
	"bufio"
	"os"
	"strings"

	context "context"

	opentracing "github.com/opentracing/opentracing-go"
	olog "github.com/opentracing/opentracing-go/log"
	"github.com/pkg/errors"
	"github.com/rai-project/config"
	"github.com/rai-project/dlframework"
	"github.com/rai-project/dlframework/framework/agent"
	"github.com/rai-project/dlframework/framework/feature"
	"github.com/rai-project/dlframework/framework/options"
	common "github.com/rai-project/dlframework/framework/predict"
	"github.com/rai-project/downloadmanager"
	gotensorrt "github.com/rai-project/go-tensorrt"
	"github.com/rai-project/image"
	"github.com/rai-project/image/types"
	"github.com/rai-project/tensorrt"
	"github.com/rai-project/tracer"
	"github.com/rai-project/tracer/ctimer"
)

// ImagePredictor ...
type ImagePredictor struct {
	common.ImagePredictor
	features  []string
	predictor *gotensorrt.Predictor
}

// New ...
func New(model dlframework.ModelManifest, opts ...options.Option) (common.Predictor, error) {
	ctx := context.Background()
	span, ctx := tracer.StartSpanFromContext(ctx, tracer.APPLICATION_TRACE, "new_predictor")
	defer span.Finish()

	modelInputs := model.GetInputs()
	if len(modelInputs) != 1 {
		return nil, errors.New("number of inputs not supported")
	}

	firstInputType := modelInputs[0].GetType()
	if strings.ToLower(firstInputType) != "image" {
		return nil, errors.New("input type not supported")
	}

	predictor := new(ImagePredictor)

	return predictor.Load(ctx, model, opts...)
}

// Download ...
func (p *ImagePredictor) Download(ctx context.Context, model dlframework.ModelManifest, opts ...options.Option) error {
	framework, err := model.ResolveFramework()
	if err != nil {
		return err
	}

	workDir, err := model.WorkDir()
	if err != nil {
		return err
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
		return err
	}

	return nil
}

// Load ...
func (p *ImagePredictor) Load(ctx context.Context, model dlframework.ModelManifest, opts ...options.Option) (common.Predictor, error) {

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

	imageDims, err := ip.GetImageDimensions()
	if err != nil {
		return nil, err
	}

	ip.Options.Append(
		options.InputNode(ip.GetInputLayerName(DefaultOutputLayerName), imageDims),
		options.OutputNode(ip.GetOutputLayerName(DefaultOutputLayerName)),
	)

	if !ip.Options.UsesGPU() {
		return nil, errors.New("TensorRT requires the GPU option to be set")
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
		tracer.APPLICATION_TRACE,
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
	span, ctx := tracer.StartSpanFromContext(ctx, tracer.APPLICATION_TRACE, "load_predictor")
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

	span.LogFields(
		olog.String("event", "creating predictor"),
	)

	opts, err := p.GetPredictionOptions(ctx)
	if err != nil {
		return err
	}

	pred, err := gotensorrt.New(
		ctx,
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
func (p *ImagePredictor) Predict(ctx context.Context, data [][]float32, opts ...options.Option) error {
	if !p.Options.UsesGPU() {
		return errors.New("TensorRT requires the GPU option to be set")
	}
	if p.TraceLevel() >= tracer.FRAMEWORK_TRACE {
		err := p.predictor.StartProfiling("tensorrt", "predict")
		if err != nil {
			log.WithError(err).WithField("framework", "tensorrt").Error("unable to start framework profiling")
		} else {
			defer func() {
				p.predictor.EndProfiling()
				profBuffer, err := p.predictor.ReadProfile()
				if err != nil {
					log.WithError(err).Error("failed to read profile")
					return
				}

				t, err := ctimer.New(profBuffer)
				if err != nil {
					log.WithError(err).WithField("json", profBuffer).Error("failed to create ctimer")
					return
				}
				t.Publish(ctx, tracer.FRAMEWORK_TRACE)

				p.predictor.DisableProfiling()
			}()
		}
	}

	var input []float32
	for _, v := range data {
		input = append(input, v...)
	}

	err := p.predictor.Predict(ctx, input)
	if err != nil {
		return err
	}

	return nil
}

// ReadPredictedFeatures ...
func (p *ImagePredictor) ReadPredictedFeatures(ctx context.Context) ([]dlframework.Features, error) {
	predictions, err := p.predictor.ReadPredictedFeatures(ctx)
	if err != nil {
		return nil, err
	}

	batchSize := int(p.BatchSize())
	length := len(predictions) / batchSize

	output := make([]dlframework.Features, batchSize)

	for ii := 0; ii < batchSize; ii++ {
		rprobs := make([]*dlframework.Feature, length)
		for jj := 0; jj < length; jj++ {
			rprobs[jj] = feature.New(
				feature.ClassificationIndex(int32(jj)),
				feature.ClassificationName(p.features[jj]),
				feature.Probability(predictions[ii*length+jj].Probability),
			)
		}
		output[ii] = rprobs
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
