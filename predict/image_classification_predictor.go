package predictor

import (
	"context"
	"fmt"
	"io/ioutil"
	"strings"

	"github.com/pkg/errors"
	"github.com/rai-project/config"
	"github.com/rai-project/dlframework"
	"github.com/rai-project/dlframework/framework/agent"
	"github.com/rai-project/dlframework/framework/options"
	common "github.com/rai-project/dlframework/framework/predictor"
	gotrt "github.com/rai-project/go-tensorrt"
	"github.com/rai-project/mxnet"
	nvidiasmi "github.com/rai-project/nvidia-smi"
	"github.com/rai-project/tracer"
	"gorgonia.org/tensor"
	gotensor "gorgonia.org/tensor"
)

// ImageClassificationPredictor
type ImageClassificationPredictor struct {
	*ImagePredictor
	probabilities interface{}
}

// NewImageClassificationPredictor initilizes the ImageClassificationPredictor.
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

// Load loads the context and actually initializes the predictor in go-tensorrt. This is different from other frameworks as the base level predictor is
// initialized in the image_predictor level. It is because TensorRT requires the input and output nodes' names when constructing the go-tensorrt predictor.
func (self *ImageClassificationPredictor) Load(ctx context.Context, modelManifest dlframework.ModelManifest, opts ...options.Option) (common.Predictor, error) {
	pred, err := self.ImagePredictor.Load(ctx, modelManifest, opts...)
	if err != nil {
		return nil, err
	}

	// Move all the loading functions from the parent level to here due to the case that initializing TensorRT model
	// requires the knowledge of input and output node info when initializing.
	if ctx != nil {
		span, _ := tracer.StartSpanFromContext(ctx, tracer.APPLICATION_TRACE, "load_predictor")
		defer span.Finish()
	}

	predOptions, err := self.GetPredictionOptions()
	if err != nil {
		return nil, errors.Wrap(err, "failed to get the prediction options")
	}

	if !nvidiasmi.HasGPU {
		panic("no GPU")
	}
	device := options.CUDA_DEVICE

	graph, err := ioutil.ReadFile(self.GetGraphPath())
	if err != nil {
		return nil, errors.Wrapf(err, "cannot read %s", self.GetGraphPath())
	}

	weights, err := ioutil.ReadFile(self.GetWeightsPath())
	if err != nil {
		return nil, errors.Wrapf(err, "cannot read %s", self.GetWeightsPath())
	}

	batchSize := self.BatchSize()

	inputName, err := self.GetInputLayerName("input_layer")
	if err != nil {
		return nil, errors.Wrap(err, "failed to get the input layer name")
	}

	inputShape, err := self.GetInputDimensions()
	if err != nil {
		return nil, errors.Wrap(err, "failed to get the input dimensions")
	}

	outputName, err := self.GetOutputLayerName("probabilities_layer")
	if err != nil {
		return nil, errors.Wrap(err, "failed to get the output layer name")
	}

	// preprocessOpts, err := self.GetPreprocessOptions()
	// if err != nil {
	// 	return nil, errors.Wrap(err, "failed to get the input preprocess options")
	// }

	// Normal input and output nodes setup.
	inputNodes := []options.Node{
		options.Node{
			Key:   inputName,
			Shape: inputShape,
			Dtype: tensor.Float32,
		},
	}

	outputNodes := []options.Node{
		options.Node{
			Key:   outputName,
			Dtype: tensor.Float32,
		},
	}

	trtPredictor, err := gotrt.New(
		ctx,
		options.WithOptions(predOptions),
		options.Device(device, 0),
		options.Graph([]byte(graph)),
		options.Weights([]byte(weights)),
		options.BatchSize(batchSize),
		options.InputNodes(inputNodes),
		options.OutputNodes(outputNodes),
	)
	if err != nil {
		panic(fmt.Sprintf("%v", err))
	}
	self.predictor = trtPredictor

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
		return nil, errors.New("cannot get the labels")
	}

	return p.CreateClassificationFeaturesFrom1D(ctx, outputs[0], labels)
}

// Modality()
func (p ImageClassificationPredictor) Modality() (dlframework.Modality, error) {
	return dlframework.ImageClassificationModality, nil
}

func init() {
	config.AfterInit(func() {
		framework := mxnet.FrameworkManifest
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
