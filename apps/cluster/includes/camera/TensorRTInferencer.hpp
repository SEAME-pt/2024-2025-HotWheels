#ifndef TENSORRTINFERENCER_HPP
#define TENSORRTINFERENCER_HPP

#pragma once

#include <QObject>
#include <NvInfer.h>
#include <cuda_runtime_api.h>
#include <opencv2/opencv.hpp>
#include <string>
#include <vector>
#include <memory>

class TensorRTInferencer : public QObject {
	Q_OBJECT

public:
	explicit TensorRTInferencer(const std::string& enginePath, QObject *parent = nullptr);
	~TensorRTInferencer();

	cv::Mat makePrediction(const cv::Mat& image);

private:
	class Logger : public nvinfer1::ILogger {
	public:
		void log(Severity severity, const char* msg) noexcept override;
	};

	Logger logger;
	std::vector<char> engineData;
	nvinfer1::IRuntime* runtime;
	nvinfer1::ICudaEngine* engine;
	nvinfer1::IExecutionContext* context;
	int inputBindingIndex;
	int outputBindingIndex;
	nvinfer1::Dims inputDims;
	nvinfer1::Dims outputDims;
	cv::Size inputSize;

	std::vector<char> readEngineFile(const std::string& enginePath);
	cv::Mat preprocessImage(const cv::Mat& image);
	std::vector<float> runInference(const cv::Mat& inputImage);
};

#endif // TENSORRTINFERENCER_HPP
