package predictor

import (
	"context"
	"fmt"
	"strings"

	"github.com/pkg/errors"
	"github.com/rai-project/config"
	"github.com/rai-project/dlframework"
	"github.com/rai-project/dlframework/framework/agent"
	"github.com/rai-project/dlframework/framework/options"
	common "github.com/rai-project/dlframework/framework/predictor"
	gotrt "github.com/rai-project/go-tensorrt"
	nvidiasmi "github.com/rai-project/nvidia-smi"
	"github.com/rai-project/tensorrt"
	"github.com/rai-project/tracer"
	gotensor "gorgonia.org/tensor"
)

// ImageClassificationPredictor
type ImageClassificationPredictor struct {
	*ImagePredictor
	probabilities interface{}
}

// NewImageClassificationPredictor initilizes the ImageClassificationPredictor
func NewImageClassificationPredictor(model dlframework.ModelManifest, opts ...options.Option) (common.Predictor, error) {
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

	predictor := new(ImageClassificationPredictor)

	return predictor.Load(ctx, model, opts...)
}

func (self *ImageClassificationPredictor) Load(ctx context.Context, modelManifest dlframework.ModelManifest, opts ...options.Option) (common.Predictor, error) {
	pred, err := self.ImagePredictor.Load(ctx, modelManifest, opts...)
	if err != nil {
		return nil, err
	}

	if ctx != nil {
		span, _ := tracer.StartSpanFromContext(ctx, tracer.APPLICATION_TRACE, "load_predictor")
		defer span.Finish()
	}

	predOptions, err := pred.GetPredictionOptions()
	if err != nil {
		return nil, errors.Wrap(err, "failed to get the prediction options")
	}

	if !nvidiasmi.HasGPU {
		panic("no GPU")
	}
	device := options.CUDA_DEVICE

	batchSize := pred.BatchSize()

	inputName, err := pred.GetInputLayerName("input_layer")
	if err != nil {
		return nil, errors.Wrap(err, "failed to get the input layer name")
	}

	inputShape, err := pred.GetInputDimensions()
	if err != nil {
		return nil, errors.Wrap(err, "failed to get the input dimensions")
	}

	outputName, err := pred.GetOutputLayerName("probabilities_layer")
	if err != nil {
		return nil, errors.Wrap(err, "failed to get the output layer name")
	}

	// preprocessOpts, err := self.GetPreprocessOptions()
	// if err != nil {
	// 	return nil, errors.Wrap(err, "failed to get the input preprocess options")
	// }

	inputNodes := []options.Node{
		options.Node{
			Key:   inputName,
			Shape: inputShape,
			Dtype: gotensor.Float32,
		},
	}

	outputNodes := []options.Node{
		options.Node{
			Key:   outputName,
			Dtype: gotensor.Float32,
		},
	}

	trtPredictor, err := gotrt.New(
		ctx,
		options.WithOptions(predOptions),
		options.Device(device, 0),
		options.Graph([]byte(pred.GetGraphPath())),
		options.Weights([]byte(pred.GetWeightsPath())),
		options.BatchSize(batchSize),
		options.InputNodes(inputNodes),
		options.OutputNodes(outputNodes),
	)
	if err != nil {
		panic(fmt.Sprintf("%v", err))
	}
	pred.predictor = trtPredictor

	p := &ImageClassificationPredictor{
		ImagePredictor: pred,
	}

	return p, nil
}

// Predict ...
func (p *ImageClassificationPredictor) Predict(ctx context.Context, data interface{}, opts ...options.Option) error {
	span, ctx := tracer.StartSpanFromContext(ctx, tracer.APPLICATION_TRACE, "predict")
	defer span.Finish()

	if data == nil {
		return errors.New("input data nil")
	}
	input, ok := data.([]*gotensor.Dense)
	if !ok {
		return errors.New("input data is not slice of go tensors")
	}

	fst := input[0]
	joined, err := fst.Concat(0, input[1:]...)
	if err != nil {
		return errors.Wrap(err, "unable to concat tensors")
	}
	joined.Reshape(append([]int{len(input)}, fst.Shape()...)...)
	inputFloat := joined.Data().([]float32)

	err = p.predictor.Predict(ctx, inputFloat)
	if err != nil {
		return errors.Wrapf(err, "failed to perform Predict")
	}

	return nil
}

// ReadPredictedFeatures ...
func (p *ImageClassificationPredictor) ReadPredictedFeatures(ctx context.Context) ([]dlframework.Features, error) {
	span, ctx := tracer.StartSpanFromContext(ctx, tracer.APPLICATION_TRACE, "read_predicted_features")
	defer span.Finish()

	outputs, err := p.predictor.ReadPredictionOutputs(ctx)
	if err != nil {
		return nil, err
	}

	labels, err := p.GetLabels()
	if err != nil {
		return nil, err
	}

	return p.CreateClassificationFeaturesFrom1D(ctx, outputs[0], labels)
}

// Modality()
func (p ImageClassificationPredictor) Modality() (dlframework.Modality, error) {
	return dlframework.ImageClassificationModality, nil
}

func init() {
	config.AfterInit(func() {
		framework := tensorrt.FrameworkManifest
		agent.AddPredictor(framework, &ImageClassificationPredictor{
			ImagePredictor: &ImagePredictor{
				ImagePredictor: common.ImagePredictor{
					Base: common.Base{
						Framework: framework,
					},
				},
			},
		})
	})
}
