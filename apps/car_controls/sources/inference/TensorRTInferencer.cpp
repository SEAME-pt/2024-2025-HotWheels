#include "TensorRTInferencer.hpp"
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <numeric>
#include <cuda_fp16.h>  // Include for __half support

// Logger callback for TensorRT to print warnings and errors
void TensorRTInferencer::Logger::log(Severity severity, const char* msg) noexcept {
	if (severity <= Severity::kWARNING)
		std::cout << msg << std::endl;
}

TensorRTInferencer::TensorRTInferencer(const std::string& enginePath) :
	runtime(nullptr), engine(nullptr), context(nullptr),
	inputBindingIndex(-1), outputBindingIndex(-1), inputSize(256, 256),
	deviceInput(nullptr), deviceOutput(nullptr), stream(nullptr),
	hostInput(nullptr), hostOutput(nullptr) {

	cudaSetDevice(0);
	engineData = readEngineFile(enginePath);

	runtime = nvinfer1::createInferRuntime(logger);
	if (!runtime) throw std::runtime_error("Failed to create TensorRT Runtime");

	engine = runtime->deserializeCudaEngine(engineData.data(), engineData.size());
	if (!engine) throw std::runtime_error("Failed to deserialize engine");

	context = engine->createExecutionContext();
	if (!context) throw std::runtime_error("Failed to create execution context");

	for (int i = 0; i < engine->getNbBindings(); i++) {
		if (engine->bindingIsInput(i)) inputBindingIndex = i;
		else outputBindingIndex = i;
	}

	if (inputBindingIndex == -1 || outputBindingIndex == -1)
		throw std::runtime_error("Could not find input and output bindings");

	inputDims = engine->getBindingDimensions(inputBindingIndex);
	outputDims = engine->getBindingDimensions(outputBindingIndex);

	if (inputDims.d[0] == -1) {
		context->setBindingDimensions(inputBindingIndex, nvinfer1::Dims4(1, inputSize.height, inputSize.width, 1));
		inputDims = context->getBindingDimensions(inputBindingIndex);
	}

	outputDims = context->getBindingDimensions(outputBindingIndex);
	for (int i = 0; i < outputDims.nbDims; i++)
		if (outputDims.d[i] < 0) throw std::runtime_error("Output shape is undefined or dynamic");

	inputElementCount = outputElementCount = 1;
	for (int i = 0; i < inputDims.nbDims; i++) inputElementCount *= static_cast<size_t>(inputDims.d[i]);
	for (int i = 0; i < outputDims.nbDims; i++) outputElementCount *= static_cast<size_t>(outputDims.d[i]);

	inputByteSize = inputElementCount * sizeof(__half);
	outputByteSize = outputElementCount * sizeof(__half);

	cudaError_t status = cudaStreamCreate(&stream);
	if (status != cudaSuccess) throw std::runtime_error("Failed to create CUDA stream: " + std::string(cudaGetErrorString(status)));

	status = cudaMalloc(&deviceInput, inputByteSize);
	if (status != cudaSuccess) throw std::runtime_error("Failed to allocate input memory on GPU: " + std::string(cudaGetErrorString(status)));

	status = cudaMalloc(&deviceOutput, outputByteSize);
	if (status != cudaSuccess) {
		cudaFree(deviceInput);
		throw std::runtime_error("Failed to allocate output memory on GPU: " + std::string(cudaGetErrorString(status)));
	}

	bindings.resize(engine->getNbBindings());
	bindings[inputBindingIndex] = deviceInput;
	bindings[outputBindingIndex] = deviceOutput;
}

void TensorRTInferencer::cleanupResources() {
	if (deviceInput) cudaFree(deviceInput);
	if (deviceOutput) cudaFree(deviceOutput);
	if (stream) cudaStreamDestroy(stream);
	deviceInput = nullptr;
	deviceOutput = nullptr;
	stream = nullptr;
}

TensorRTInferencer::~TensorRTInferencer() {
	if (hostInput) cudaFreeHost(hostInput);
	if (hostOutput) cudaFreeHost(hostOutput);
	cleanupResources();
	if (context) context->destroy();
	if (engine) engine->destroy();
	if (runtime) runtime->destroy();
}

std::vector<char> TensorRTInferencer::readEngineFile(const std::string& enginePath) {
	std::ifstream file(enginePath, std::ios::binary | std::ios::ate);
	if (!file.good()) throw std::runtime_error("Engine file not found: " + enginePath);
	size_t size = file.tellg();
	file.seekg(0, std::ios::beg);
	std::vector<char> buffer(size);
	if (!file.read(buffer.data(), size)) throw std::runtime_error("Failed to read engine file");
	return buffer;
}

cv::cuda::GpuMat TensorRTInferencer::preprocessImage(const cv::cuda::GpuMat& gpuImage) {
	if (gpuImage.empty()) throw std::runtime_error("Input image is empty");

	cv::cuda::GpuMat gpuGray, gpuResized, gpuFloat;
	if (gpuImage.channels() > 1) cv::cuda::cvtColor(gpuImage, gpuGray, cv::COLOR_BGR2GRAY);
	else gpuGray = gpuImage;

	cv::cuda::resize(gpuGray, gpuResized, inputSize, 0, 0, cv::INTER_LINEAR);
	gpuResized.convertTo(gpuFloat, CV_16F, 1.0 / 255.0);
	return gpuFloat;
}

void TensorRTInferencer::runInference(const cv::cuda::GpuMat& gpuInput) {
	if (gpuInput.rows != inputSize.height || gpuInput.cols != inputSize.width)
		throw std::runtime_error("Input image dimensions do not match expected dimensions");

	cudaError_t err = cudaMemcpy2DAsync(
		deviceInput,
		inputSize.width * sizeof(__half),
		reinterpret_cast<const void*>(gpuInput.ptr<__half>()),
		gpuInput.step,
		inputSize.width * sizeof(__half),
		inputSize.height,
		cudaMemcpyDeviceToDevice,
		stream);

	if (err != cudaSuccess) throw std::runtime_error("cudaMemcpy2DAsync failed: " + std::string(cudaGetErrorString(err)));
	if (!context->enqueueV2(bindings.data(), stream, nullptr)) throw std::runtime_error("TensorRT inference execution failed");
}

cv::cuda::GpuMat TensorRTInferencer::makePrediction(const cv::cuda::GpuMat& gpuImage) {
	cv::cuda::GpuMat gpuInputFloat = preprocessImage(gpuImage);
	runInference(gpuInputFloat);

	int height = outputDims.d[1];
	int width = outputDims.d[2];

	if (outputMaskGpu.empty() || outputMaskGpu.rows != height || outputMaskGpu.cols != width) {
		outputMaskGpu = cv::cuda::GpuMat(height, width, CV_16F);
	}

	cudaMemcpy2DAsync(
		reinterpret_cast<void*>(outputMaskGpu.ptr<__half>()), outputMaskGpu.step,
		deviceOutput, width * sizeof(__half),
		width * sizeof(__half), height,
		cudaMemcpyDeviceToDevice, stream);

	return outputMaskGpu;
}
