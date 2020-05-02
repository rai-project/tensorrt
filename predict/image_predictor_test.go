package predictor

import (
	"context"
	"os"
	"path/filepath"
	"testing"

	"github.com/rai-project/dlframework/framework/options"
	raiimage "github.com/rai-project/image"
	"github.com/rai-project/image/types"
	trt "github.com/rai-project/tensorrt"
	"github.com/stretchr/testify/assert"
	gotensor "gorgonia.org/tensor"
)

// convert go RGB Image to 1D normalized RGB array
func cvtRGBImageToNCHW1DBGRArray(in types.Image, mean []float32, scale []float32) ([]float32, error) {
	channels := in.Channels()
	height := in.Bounds().Dy()
	width := in.Bounds().Dx()
	stride := width * height // image size per channel
	pix := in.Pixels()
	out := make([]float32, channels*height*width)

	for y := 0; y < height; y++ {
		for x := 0; x < width; x++ {
			offset := y*stride + x*channels
			rgb := pix[offset : offset+channels]
			r, g, b := rgb[0], rgb[1], rgb[2]
			out[0*stride+y*width+x] = (float32(b) - mean[0]) / scale[0]
			out[1*stride+y*width+x] = (float32(g) - mean[1]) / scale[1]
			out[2*stride+y*width+x] = (float32(r) - mean[2]) / scale[2]
		}
	}

	return out, nil
}

func TestNewImageClassificationPredictor(t *testing.T) {
	trt.Register()
	model, err := trt.FrameworkManifest.FindModel("BVLC-GoogLeNet:1.0")
	assert.NoError(t, err)
	assert.NotEmpty(t, model)

	predictor, err := NewImageClassificationPredictor(*model)
	assert.NoError(t, err)
	assert.NotEmpty(t, predictor)

	defer predictor.Close()

	imgPredictor, ok := predictor.(*ImageClassificationPredictor)
	assert.True(t, ok)

	assert.NotEmpty(t, imgPredictor)
}

func TestImageClassification(t *testing.T) {
	trt.Register()
	model, err := trt.FrameworkManifest.FindModel("BVLC-GoogLeNet:1.0")
	assert.NoError(t, err)
	assert.NotEmpty(t, model)

	device := options.CUDA_DEVICE

	batchSize := 1
	ctx := context.Background()
	opts := options.New(options.Context(ctx),
		options.Device(device, 0),
		options.BatchSize(batchSize))

	predictor, err := NewImageClassificationPredictor(*model, options.WithOptions(opts))
	assert.NoError(t, err)
	assert.NotEmpty(t, predictor)
	defer predictor.Close()

	preprocessOpts, err := predictor.GetPreprocessOptions()
	assert.NoError(t, err)
	channels := preprocessOpts.Dims[0]
	height := preprocessOpts.Dims[1]
	width := preprocessOpts.Dims[2]
	mode := preprocessOpts.ColorMode

	imgOpts := []raiimage.Option{
		raiimage.Mode(mode),
		raiimage.Width(width),
		raiimage.Height(height),
		raiimage.ResizeAlgorithm(types.ResizeAlgorithmLinear),
	}

	imgDir, _ := filepath.Abs("./_fixtures")
	imgPath := filepath.Join(imgDir, "platypus.jpg")
	r, err := os.Open(imgPath)
	if err != nil {
		panic(err)
	}

	img, err := raiimage.Read(r, imgOpts...)
	if err != nil {
		panic(err)
	}

	input := make([]*gotensor.Dense, batchSize)
	imgFloats, err := cvtRGBImageToNCHW1DBGRArray(img, preprocessOpts.MeanImage, preprocessOpts.Scale)
	if err != nil {
		panic(err)
	}

	for ii := 0; ii < batchSize; ii++ {
		input[ii] = gotensor.New(
			gotensor.WithShape(height, width, channels),
			gotensor.WithBacking(imgFloats),
		)
	}

	err = predictor.Predict(ctx, input)
	assert.NoError(t, err)
	if err != nil {
		return
	}

	pred, err := predictor.ReadPredictedFeatures(ctx)
	assert.NoError(t, err)
	if err != nil {
		return
	}
	assert.InDelta(t, float32(0.967381), pred[0][0].GetProbability(), 0.001)
	assert.Equal(t, int32(103), pred[0][0].GetClassification().GetIndex())
}

// func TestInstanceSegmentation(t *testing.T) {
// 	trt.Register()
// 	model, err := trt.FrameworkManifest.FindModel("mask_rcnn_inception_v2_coco:1.0")
// 	assert.NoError(t, err)
// 	assert.NotEmpty(t, model)

// 	device := options.CPU_DEVICE
// 	if nvidiasmi.HasGPU {
// 		device = options.CUDA_DEVICE
// 	}

// 	batchSize := 1
// 	ctx := context.Background()
// 	opts := options.New(options.Context(ctx),
// 		options.Device(device, 0),
// 		options.BatchSize(batchSize))

// 	predictor, err := NewInstanceSegmentationPredictor(*model, options.WithOptions(opts))
// 	assert.NoError(t, err)
// 	assert.NotEmpty(t, predictor)
// 	defer predictor.Close()

// 	imgDir, _ := filepath.Abs("./_fixtures")
// 	imgPath := filepath.Join(imgDir, "lane_control.jpg")
// 	r, err := os.Open(imgPath)
// 	if err != nil {
// 		panic(err)
// 	}
// 	img, err := raiimage.Read(r)
// 	if err != nil {
// 		panic(err)
// 	}

// 	height := img.Bounds().Dy()
// 	width := img.Bounds().Dx()
// 	channels := 3
// 	input := make([]*gotensor.Dense, batchSize)
// 	imgBytes := img.(*types.RGBImage).Pix

// 	for ii := 0; ii < batchSize; ii++ {
// 		input[ii] = gotensor.New(
// 			gotensor.WithShape(height, width, channels),
// 			gotensor.WithBacking(imgBytes),
// 		)
// 	}

// 	err = predictor.Predict(ctx, input)
// 	assert.NoError(t, err)
// 	if err != nil {
// 		return
// 	}

// 	pred, err := predictor.ReadPredictedFeatures(ctx)
// 	assert.NoError(t, err)
// 	if err != nil {
// 		return
// 	}

// 	assert.InDelta(t, float32(0.998607), pred[0][0].GetProbability(), 0.001)
// }

// func TestObjectDetection(t *testing.T) {
// 	trt.Register()
// 	model, err := trt.FrameworkManifest.FindModel("SSD_300_VGG16_Atrous_COCO:1.0")
// 	assert.NoError(t, err)
// 	assert.NotEmpty(t, model)

// 	device := options.CPU_DEVICE
// 	if nvidiasmi.HasGPU {
// 		device = options.CUDA_DEVICE
// 	}

// 	batchSize := 2
// 	ctx := context.Background()
// 	opts := options.New(options.Context(ctx),
// 		options.Device(device, 0),
// 		options.BatchSize(batchSize))

// 	predictor, err := NewObjectDetectionPredictor(*model, options.WithOptions(opts))
// 	assert.NoError(t, err)
// 	assert.NotEmpty(t, predictor)
// 	defer predictor.Close()

// 	preprocessOpts, err := predictor.GetPreprocessOptions()
// 	assert.NoError(t, err)
// 	channels := preprocessOpts.Dims[0]
// 	height := preprocessOpts.Dims[1]
// 	width := preprocessOpts.Dims[2]
// 	mode := preprocessOpts.ColorMode

// 	imgOpts := []raiimage.Option{
// 		raiimage.Mode(mode),
// 		raiimage.Width(width),
// 		raiimage.Height(height),
// 		raiimage.ResizeAlgorithm(types.ResizeAlgorithmLinear),
// 	}

// 	imgDir, _ := filepath.Abs("./_fixtures")
// 	imgPath := filepath.Join(imgDir, "3dogs.jpg")
// 	r, err := os.Open(imgPath)
// 	if err != nil {
// 		panic(err)
// 	}

// 	img, err := raiimage.Read(r, imgOpts...)
// 	if err != nil {
// 		panic(err)
// 	}

// 	input := make([]*gotensor.Dense, batchSize)
// 	imgFloats, err := normalizeImageCHW(img, preprocessOpts.MeanImage, preprocessOpts.Scale)
// 	if err != nil {
// 		panic(err)
// 	}

// 	for ii := 0; ii < batchSize; ii++ {
// 		input[ii] = gotensor.New(
// 			gotensor.WithShape(height, width, channels),
// 			gotensor.WithBacking(imgFloats),
// 		)
// 	}

// 	err = predictor.Predict(ctx, input)
// 	assert.NoError(t, err)
// 	if err != nil {
// 		return
// 	}

// 	pred, err := predictor.ReadPredictedFeatures(ctx)
// 	assert.NoError(t, err)
// 	if err != nil {
// 		return
// 	}
// 	assert.InDelta(t, float32(0.996272), pred[0][0].GetProbability(), 0.001)
// 	assert.Equal(t, int32(11), pred[0][0].GetBoundingBox().GetIndex())
// }

// func max(x, y int) int {
// 	if x < y {
// 		return y
// 	}
// 	return x
// }

// func TestSemanticSegmentation(t *testing.T) {
// 	trt.Register()
// 	model, err := trt.FrameworkManifest.FindModel("DeepLabv3_PASCAL_VOC_Train_Aug:1.0")
// 	assert.NoError(t, err)
// 	assert.NotEmpty(t, model)

// 	device := options.CPU_DEVICE
// 	if nvidiasmi.HasGPU {
// 		device = options.CUDA_DEVICE
// 	}

// 	batchSize := 1
// 	ctx := context.Background()
// 	opts := options.New(options.Context(ctx),
// 		options.Device(device, 0),
// 		options.BatchSize(batchSize))

// 	predictor, err := NewSemanticSegmentationPredictor(*model, options.WithOptions(opts))
// 	assert.NoError(t, err)
// 	assert.NotEmpty(t, predictor)
// 	defer predictor.Close()

// 	imgDir, _ := filepath.Abs("./_fixtures")
// 	imgPath := filepath.Join(imgDir, "lane_control.jpg")
// 	r, err := os.Open(imgPath)
// 	if err != nil {
// 		panic(err)
// 	}
// 	img, err := raiimage.Read(r)
// 	if err != nil {
// 		panic(err)
// 	}

// 	height := img.Bounds().Dy()
// 	width := img.Bounds().Dx()
// 	channels := 3
// 	inputSize := 513
// 	resizeRatio := float32(inputSize) / float32(max(width, height))
// 	targetWidth := int(resizeRatio * float32(width))
// 	targetHeight := int(resizeRatio * float32(height))
// 	resized, err := raiimage.Resize(img, raiimage.Resized(targetHeight, targetWidth))
// 	if err != nil {
// 		panic(err)
// 	}
// 	input := make([]*gotensor.Dense, batchSize)
// 	imgBytes := resized.(*types.RGBImage).Pix
// 	for ii := 0; ii < batchSize; ii++ {
// 		input[ii] = gotensor.New(
// 			gotensor.WithShape(targetHeight, targetWidth, channels),
// 			gotensor.WithBacking(imgBytes),
// 		)
// 	}

// 	err = predictor.Predict(ctx, input)
// 	assert.NoError(t, err)
// 	if err != nil {
// 		return
// 	}

// 	pred, err := predictor.ReadPredictedFeatures(ctx)
// 	assert.NoError(t, err)
// 	if err != nil {
// 		return
// 	}

// 	sseg := pred[0][0].GetSemanticSegment()
// 	intMask := sseg.GetIntMask()

// 	assert.Equal(t, int32(7), intMask[72122])
// }
