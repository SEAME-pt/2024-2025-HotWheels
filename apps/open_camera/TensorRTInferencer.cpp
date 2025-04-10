#include "TensorRTInferencer.hpp"
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <numeric>

void TensorRTInferencer::Logger::log(Severity severity, const char* msg) noexcept {
	if (severity <= Severity::kWARNING)
		std::cout << msg << std::endl;
}

TensorRTInferencer::TensorRTInferencer(const std::string& enginePath) :
	runtime(nullptr),
	engine(nullptr),
	context(nullptr),
	inputBindingIndex(-1),
	outputBindingIndex(-1),
	inputSize(256, 256) {

	cudaSetDevice(0);

	engineData = readEngineFile(enginePath);

	runtime = nvinfer1::createInferRuntime(logger);
	if (!runtime) {
		throw std::runtime_error("Failed to create TensorRT Runtime");
	}

	engine = runtime->deserializeCudaEngine(engineData.data(), engineData.size());
	if (!engine) {
		throw std::runtime_error("Failed to deserialize engine");
	}

	context = engine->createExecutionContext();
	if (!context) {
		throw std::runtime_error("Failed to create execution context");
	}

	for (int i = 0; i < engine->getNbBindings(); i++) {
		if (engine->bindingIsInput(i)) {
			inputBindingIndex = i;
		} else {
			outputBindingIndex = i;
		}
	}

	if (inputBindingIndex == -1 || outputBindingIndex == -1) {
		throw std::runtime_error("Could not find input and output bindings");
	}

	inputDims = engine->getBindingDimensions(inputBindingIndex);
	outputDims = engine->getBindingDimensions(outputBindingIndex);

	if (inputDims.d[0] == -1) {
		nvinfer1::Dims4 explicitDims(1, inputSize.height, inputSize.width, 1);
		context->setBindingDimensions(inputBindingIndex, explicitDims);
		inputDims = context->getBindingDimensions(inputBindingIndex);
	}

	outputDims = context->getBindingDimensions(outputBindingIndex);

	for (int i = 0; i < outputDims.nbDims; i++) {
		if (outputDims.d[i] < 0) {
			throw std::runtime_error("Output shape is undefined or dynamic");
		}
	}
}

TensorRTInferencer::~TensorRTInferencer() {
	if (context) {
		context->destroy();
	}
	if (engine) {
		engine->destroy();
	}
	if (runtime) {
		runtime->destroy();
	}
}

std::vector<char> TensorRTInferencer::readEngineFile(const std::string& enginePath) {
	std::ifstream file(enginePath, std::ios::binary | std::ios::ate);
	if (!file.good()) {
		throw std::runtime_error("Engine file not found: " + enginePath);
	}

	size_t size = file.tellg();
	file.seekg(0, std::ios::beg);

	std::vector<char> buffer(size);
	if (!file.read(buffer.data(), size)) {
		throw std::runtime_error("Failed to read engine file");
	}

	return buffer;
}

cv::Mat TensorRTInferencer::preprocessImage(const cv::Mat& image) {
	if (image.empty()) {
		throw std::runtime_error("Input image is empty");
	}

	cv::Mat grayImg;
	if (image.channels() > 1) {
		cv::cvtColor(image, grayImg, cv::COLOR_BGR2GRAY);
	} else {
		grayImg = image.clone();
	}

	cv::Mat resizedImg;
	cv::resize(grayImg, resizedImg, inputSize, 0, 0, cv::INTER_LANCZOS4);

	cv::Mat floatImg;
	resizedImg.convertTo(floatImg, CV_32F, 1.0/255.0);

	return floatImg;
}

std::vector<float> TensorRTInferencer::runInference(const cv::Mat& inputImage) {
	size_t inputElementCount = 1;
	for (int i = 0; i < inputDims.nbDims; i++) {
		inputElementCount *= static_cast<size_t>(inputDims.d[i]);
	}
	size_t inputSize = inputElementCount * sizeof(float);

	size_t outputElementCount = 1;
	for (int i = 0; i < outputDims.nbDims; i++) {
		outputElementCount *= static_cast<size_t>(outputDims.d[i]);
	}
	size_t outputSize = outputElementCount * sizeof(float);

	std::vector<float> inputBuffer(inputElementCount);
	std::vector<float> outputBuffer(outputElementCount);

	float* hostDataBuffer = inputBuffer.data();
	for (int h = 0; h < this->inputSize.height; h++) {
		for (int w = 0; w < this->inputSize.width; w++) {
			hostDataBuffer[h * this->inputSize.width * 1 + w * 1 + 0] = inputImage.at<float>(h, w);
		}
	}

	void* deviceInput = nullptr;
	void* deviceOutput = nullptr;
	cudaError_t cudaStatus;

	try {
		cudaStatus = cudaMalloc(&deviceInput, inputSize);
		if (cudaStatus != cudaSuccess) {
			throw std::runtime_error(std::string("CUDA Memory Allocation Error for input: ") +
									cudaGetErrorString(cudaStatus));
		}

		cudaStatus = cudaMalloc(&deviceOutput, outputSize);
		if (cudaStatus != cudaSuccess) {
			throw std::runtime_error(std::string("CUDA Memory Allocation Error for output: ") +
									cudaGetErrorString(cudaStatus));
		}

		cudaStatus = cudaMemcpy(deviceInput, inputBuffer.data(), inputSize, cudaMemcpyHostToDevice);
		if (cudaStatus != cudaSuccess) {
			throw std::runtime_error(std::string("CUDA Memcpy Error: ") +
									cudaGetErrorString(cudaStatus));
		}

		std::vector<void*> bindings(engine->getNbBindings(), nullptr);
		bindings[inputBindingIndex] = deviceInput;
		bindings[outputBindingIndex] = deviceOutput;

		bool status = context->executeV2(bindings.data());
		if (!status) {
			throw std::runtime_error("Failed to execute inference");
		}

		cudaStatus = cudaMemcpy(outputBuffer.data(), deviceOutput, outputSize, cudaMemcpyDeviceToHost);
		if (cudaStatus != cudaSuccess) {
			throw std::runtime_error(std::string("CUDA Memcpy Error (device to host): ") +
									cudaGetErrorString(cudaStatus));
		}
	}
	catch (const std::exception& e) {
		if (deviceInput) cudaFree(deviceInput);
		if (deviceOutput) cudaFree(deviceOutput);
		throw;
	}

	cudaFree(deviceInput);
	cudaFree(deviceOutput);

	return outputBuffer;
}

cv::Mat TensorRTInferencer::makePrediction(const cv::Mat& image) {
	cv::Mat inputImage = preprocessImage(image);
	std::vector<float> outputBuffer = runInference(inputImage);

	int height = outputDims.d[1];
	int width = outputDims.d[2];

	cv::Mat outputMask(height, width, CV_32F);

	for (int h = 0; h < height; h++) {
		for (int w = 0; w < width; w++) {
			int idx = h * width * 1 + w * 1 + 0;
			outputMask.at<float>(h, w) = outputBuffer[idx];
		}
	}

	return outputMask;
}
