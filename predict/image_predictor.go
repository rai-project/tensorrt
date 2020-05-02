package predictor

import (
	context "context"

	opentracing "github.com/opentracing/opentracing-go"
	olog "github.com/opentracing/opentracing-go/log"
	"github.com/pkg/errors"
	"github.com/rai-project/dlframework"
	"github.com/rai-project/dlframework/framework/options"
	common "github.com/rai-project/dlframework/framework/predictor"
	"github.com/rai-project/downloadmanager"
	gotensorrt "github.com/rai-project/go-tensorrt"
)

// ImagePredictor ...
type ImagePredictor struct {
	common.ImagePredictor
	predictor *gotensorrt.Predictor
}

func (p *ImagePredictor) GetOutputLayerName(layer string) (string, error) {
	model := p.Model
	modelOutput := model.GetOutput()
	typeParameters := modelOutput.GetParameters()
	name, err := p.GetTypeParameter(typeParameters, layer)
	if err != nil {
		return "", err
	}
	return name, nil
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
				WorkDir:   workDir,
				Options:   options.New(opts...),
			},
		},
	}

	if err = ip.download(ctx); err != nil {
		return err
	}

	return nil
}

func (p *ImagePredictor) download(ctx context.Context) error {
	span, ctx := opentracing.StartSpanFromContext(
		ctx,
		"download",
		opentracing.Tags{
			"graph_url":           p.GetGraphUrl(),
			"target_graph_file":   p.GetGraphPath(),
			"weights_url":         p.GetWeightsUrl(),
			"target_weights_file": p.GetWeightsPath(),
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
	} else {
		span.LogFields(
			olog.String("event", "download model graph"),
		)
		_, _, err := downloadmanager.DownloadFile(
			p.GetGraphUrl(),
			p.GetGraphPath(),
			downloadmanager.MD5Sum(p.GetGraphChecksum()),
		)
		if err != nil {
			return err
		}

		span.LogFields(
			olog.String("event", "download model weights"),
		)
		_, _, err = downloadmanager.DownloadFile(
			p.GetWeightsUrl(),
			p.GetWeightsPath(),
			downloadmanager.MD5Sum(p.GetWeightsChecksum()),
		)
		if err != nil {
			return err
		}
	}

	if p.GetFeaturesUrl() != "" {
		span.LogFields(
			olog.String("event", "download features"),
		)
		_, _, err := downloadmanager.DownloadFile(
			p.GetFeaturesUrl(),
			p.GetFeaturesPath(),
			downloadmanager.MD5Sum(p.GetFeaturesChecksum()),
		)
		if err != nil {
			return err
		}
	}

	return nil
}

// Load ...
func (p *ImagePredictor) Load(ctx context.Context, model dlframework.ModelManifest, opts ...options.Option) (*ImagePredictor, error) {
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
				WorkDir:   workDir,
				Options:   options.New(opts...),
			},
		},
	}

	if err = ip.download(ctx); err != nil {
		return nil, err
	}

	// if err = ip.loadPredictor(ctx); err != nil {
	// 	return nil, err
	// }

	return ip, nil
}

// func (p *ImagePredictor) loadPredictor(ctx context.Context) error {
// 	if ctx != nil {
// 		span, ctx := tracer.StartSpanFromContext(ctx, tracer.APPLICATION_TRACE, "load_predictor")
// 		defer span.Finish()
// 	}

// 	graph, err := ioutil.ReadFile(p.GetGraphPath())
// 	if err != nil {
// 		return errors.Wrapf(err, "cannot read %s", p.GetGraphPath())
// 	}

// 	weights, err := ioutil.ReadFile(p.GetWeightsPath())
// 	if err != nil {
// 		return errors.Wrapf(err, "cannot read %s", p.GetWeightsPath())
// 	}

// 	opts, err := p.GetPredictionOptions()
// 	if err != nil {
// 		return errors.Wrap(err, "failed to get the prediction options")
// 	}

// 	preprocessOpts, err := p.GetPreprocessOptions()
// 	if err != nil {
// 		return errors.Wrap(err, "failed to get the input preprocess options")
// 	}

// 	batchSize := p.BatchSize()

// 	return nil
// }

// Close ...
func (p *ImagePredictor) Close() error {
	if p == nil {
		return nil
	}
	if p.predictor != nil {
		p.predictor.Close()
	}
	return nil
}

// // New ...
// func New(model dlframework.ModelManifest, opts ...options.Option) (common.Predictor, error) {
// 	ctx := context.Background()
// 	span, ctx := tracer.StartSpanFromContext(ctx, tracer.APPLICATION_TRACE, "new_predictor")
// 	defer span.Finish()

// 	modelInputs := model.GetInputs()
// 	if len(modelInputs) != 1 {
// 		return nil, errors.New("number of inputs not supported")
// 	}

// 	firstInputType := modelInputs[0].GetType()
// 	if strings.ToLower(firstInputType) != "image" {
// 		return nil, errors.New("input type not supported")
// 	}

// 	predictor := new(ImagePredictor)

// 	return predictor.Load(ctx, model, opts...)
// }

// // GetPreprocessOptions ...
// func (p *ImagePredictor) GetPreprocessOptions(ctx context.Context) (common.PreprocessOptions, error) {
// 	mean, err := p.GetMeanImage()
// 	if err != nil {
// 		return common.PreprocessOptions{}, err
// 	}

// 	scale, err := p.GetScale()
// 	if err != nil {
// 		return common.PreprocessOptions{}, err
// 	}

// 	imageDims, err := p.GetImageDimensions()
// 	if err != nil {
// 		return common.PreprocessOptions{}, err
// 	}

// 	return common.PreprocessOptions{
// 		Context:   ctx,
// 		MeanImage: mean,
// 		Scale:     scale,
// 		Size:      []int{int(imageDims[1]), int(imageDims[2])},
// 		ColorMode: types.BGRMode,
// 		Layout:    image.CHWLayout,
// 	}, nil
// }

// // Predict ...
// func (p *ImagePredictor) Predict(ctx context.Context, data [][]float32, opts ...options.Option) error {
// 	if !p.Options.UsesGPU() {
// 		return errors.New("TensorRT requires the GPU option to be set")
// 	}
// 	if p.TraceLevel() >= tracer.FRAMEWORK_TRACE {
// 		err := p.predictor.StartProfiling("tensorrt", "predict")
// 		if err != nil {
// 			log.WithError(err).WithField("framework", "tensorrt").Error("unable to start framework profiling")
// 		} else {
// 			defer func() {
// 				p.predictor.EndProfiling()
// 				profBuffer, err := p.predictor.ReadProfile()
// 				if err != nil {
// 					log.WithError(err).Error("failed to read profile")
// 					return
// 				}

// 				t, err := ctimer.New(profBuffer)
// 				if err != nil {
// 					log.WithError(err).WithField("json", profBuffer).Error("failed to create ctimer")
// 					return
// 				}
// 				t.Publish(ctx, tracer.FRAMEWORK_TRACE)

// 				p.predictor.DisableProfiling()
// 			}()
// 		}
// 	}

// 	var input []float32
// 	for _, v := range data {
// 		input = append(input, v...)
// 	}

// 	err := p.predictor.Predict(ctx, input)
// 	if err != nil {
// 		return err
// 	}

// 	return nil
// }

// // ReadPredictedFeatures ...
// func (p *ImagePredictor) ReadPredictedFeatures(ctx context.Context) ([]dlframework.Features, error) {
// 	span, ctx := tracer.StartSpanFromContext(ctx, tracer.APPLICATION_TRACE, "read_predicted_features")
// 	defer span.Finish()

// 	output, err := p.predictor.ReadPredictionOutput(ctx)
// 	if err != nil {
// 		return nil, err
// 	}

// 	return p.CreatePredictedFeatures(ctx, output, p.labels)
// }

// // Reset ...
// func (p *ImagePredictor) Reset(ctx context.Context) error {

// 	return nil
// }

// func init() {
// 	config.AfterInit(func() {
// 		framework := tensorrt.FrameworkManifest
// 		agent.AddPredictor(framework, &ImagePredictor{
// 			ImagePredictor: common.ImagePredictor{
// 				Base: common.Base{
// 					Framework: framework,
// 				},
// 			},
// 		})
// 	})
// }
