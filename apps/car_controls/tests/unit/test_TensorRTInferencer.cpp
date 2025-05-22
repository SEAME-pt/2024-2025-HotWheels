// test_TensorRTInferencer.cpp

#include <gtest/gtest.h>
#include "../includes/inference/TensorRTInferencer.hpp"

#include <opencv2/core/cuda.hpp>
#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>

TEST(TensorRTInferencerTest, CanReadEngineFile) {
    TensorRTInferencer inferencer("/home/hotweels/dev/model_loader/models/model.engine");  // Replace with a real path
    ASSERT_TRUE(true);  // If it didn't throw, it's OK for this smoke test
}

TEST(TensorRTInferencerTest, PreprocessImageGrayscale) {
    // Create a fake BGR image on CPU
    cv::Mat cpuImg(208, 208, CV_8UC3, cv::Scalar(128, 128, 128));
    cv::cuda::GpuMat gpuImg;
    gpuImg.upload(cpuImg);  // Upload to GPU

    TensorRTInferencer inferencer("/home/hotweels/dev/model_loader/models/model.engine");

    cv::cuda::GpuMat processed = inferencer.preprocessImage(gpuImg);

    EXPECT_EQ(processed.type(), CV_32F);
    EXPECT_EQ(processed.size(), cv::Size(208, 208));
}

TEST(TensorRTInferencerTest, ThrowsOnEmptyImage) {
    cv::cuda::GpuMat emptyImg;

    TensorRTInferencer inferencer("/home/hotweels/dev/model_loader/models/model.engine");

    EXPECT_THROW(inferencer.preprocessImage(emptyImg), std::runtime_error);
}
