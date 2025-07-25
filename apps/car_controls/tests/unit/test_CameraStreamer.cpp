#include <gtest/gtest.h>
#include <opencv2/core.hpp>
#include <thread>
#include "../../includes/inference/CameraStreamer.hpp"

// We’ll skip actual inference because those inferencer classes are internal
// and not mockable without modifying CameraStreamer.

/* class CameraStreamerBasicTest : public ::testing::Test {
protected:
    void SetUp() override {
        // Use scale factor only, since we can't inject models
        streamer = std::make_unique<CameraStreamer>(0.5);
    }

    void TearDown() override {
        if (streamer) {
            streamer->stop();  // Just in case
        }
    }

    std::unique_ptr<CameraStreamer> streamer;
};

TEST_F(CameraStreamerBasicTest, StartsAndStopsWithoutCrash) {
    ASSERT_NO_THROW({
        streamer->start();
        std::this_thread::sleep_for(std::chrono::milliseconds(200));
        streamer->stop();
    });
}

TEST_F(CameraStreamerBasicTest, DestructorCleansUpProperly) {
    streamer->start();
    std::this_thread::sleep_for(std::chrono::milliseconds(200));
    streamer.reset();  // Calls destructor
    SUCCEED();  // No crash means success
}

TEST_F(CameraStreamerBasicTest, RunsCaptureLoopBriefly) {
    streamer->start();
    std::this_thread::sleep_for(std::chrono::seconds(1));  // Give time for captureLoop
    streamer->stop();
    SUCCEED();
} */
