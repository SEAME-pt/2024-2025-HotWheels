#include "CameraStreamer.hpp"

// Constructor
CameraStreamer::CameraStreamer(TensorRTInferencer& infer, double scale, const std::string& win_name, bool show_orig)
    : scale_factor(scale), window_name(win_name), inferencer(infer), show_original(show_orig) {

    std::string pipeline = "nvarguscamerasrc ! video/x-raw(memory:NVMM), width=1280, height=720, format=(string)NV12, framerate=60/1 ! nvvidconv ! video/x-raw, format=(string)BGRx ! videoconvert ! video/x-raw, format=(string)BGR ! appsink";
    cap.open(pipeline, cv::CAP_GSTREAMER);

    if (!cap.isOpened()) {
        std::cerr << "Error: Could not open CSI camera" << std::endl;
        exit(-1);
    }
}

// Start streaming with inference
/* void CameraStreamer::start() {
    cv::Mat frame;

    while (true) {
        cap >> frame;
        if (frame.empty())
            break;

	//starts here
	cv::Mat cameraMatrix, distCoeffs;
	cv::FileStorage fs("camera_calibration.yml", cv::FileStorage::READ);
	fs["camera_matrix"] >> cameraMatrix;
	fs["distortion_coefficients"] >> distCoeffs;
	fs.release();

	cv::Mat undistorted;
	cv::undistort(frame, undistorted, cameraMatrix, distCoeffs);
	// ends here

        // Run inference on the frame
        cv::Mat prediction_mask = inferencer.makePrediction(undistorted);

        // Convert prediction mask to visualization (assuming values between 0-1)
        cv::Mat visualization;
        prediction_mask.convertTo(visualization, CV_8U, 255.0);

        // Apply color map for better visualization
        cv::Mat colorized_mask;
        cv::applyColorMap(visualization, colorized_mask, cv::COLORMAP_JET);

        // Resize for display
        cv::Mat resized_mask;
        cv::resize(colorized_mask, resized_mask, cv::Size(frame.cols * scale_factor, frame.rows * scale_factor), 0, 0, cv::INTER_LINEAR);

        if (show_original) {
            // Show both original and prediction
            cv::Mat resized_frame;
            cv::resize(frame, resized_frame, cv::Size(frame.cols * scale_factor, frame.rows * scale_factor), 0, 0, cv::INTER_LINEAR);

            // Make sure both images have the same type before concatenation
            if (resized_frame.type() != resized_mask.type()) {
                cv::cvtColor(resized_mask, resized_mask, cv::COLOR_BGR2BGRA);
                cv::cvtColor(resized_frame, resized_frame, cv::COLOR_BGR2BGRA);
            }

            // Make sure both images have the same size
            if (resized_frame.size() != resized_mask.size()) {
                cv::resize(resized_mask, resized_mask, resized_frame.size());
            }

            // Create a combined view
            try {
                cv::Mat combined;
                cv::hconcat(resized_frame, resized_mask, combined);
                cv::imshow(window_name, combined);
            } catch (const cv::Exception& e) {
                // If concatenation fails, show images in separate windows
                std::cerr << "Warning: Could not concatenate images: " << e.what() << std::endl;
                cv::imshow(window_name + " - Original", resized_frame);
                cv::imshow(window_name + " - Prediction", resized_mask);
            }
        } else {
            // Show only the prediction mask
            cv::imshow(window_name, resized_mask);
        }

        if (cv::waitKey(1) == 27)
            break; // Exit on ESC key
    }
} */

// Add to your class constructor or initialization function
void CameraStreamer::initUndistortMaps() {
    cv::Mat cameraMatrix, distCoeffs;
    cv::FileStorage fs("camera_calibration.yml", cv::FileStorage::READ);
    fs["camera_matrix"] >> cameraMatrix;
    fs["distortion_coefficients"] >> distCoeffs;
    fs.release();

    // Create undistortion maps
    cv::Mat mapx, mapy;
    cv::initUndistortRectifyMap(cameraMatrix, distCoeffs, cv::Mat(), cameraMatrix,
                               cv::Size(cap.get(cv::CAP_PROP_FRAME_WIDTH), cap.get(cv::CAP_PROP_FRAME_HEIGHT)),
                               CV_32FC1, mapx, mapy);

    // Upload maps to GPU (store as class members)
    d_mapx.upload(mapx);
    d_mapy.upload(mapy);
}

void CameraStreamer::start() {
    initUndistortMaps(); // Initialize undistortion maps
    cv::Mat frame;

    // Load camera calibration data once, not in the loop
    cv::Mat cameraMatrix, distCoeffs;
    cv::FileStorage fs("camera_calibration.yml", cv::FileStorage::READ);
    fs["camera_matrix"] >> cameraMatrix;
    fs["distortion_coefficients"] >> distCoeffs;
    fs.release();

    // Create CUDA stream for asynchronous operations
    cv::cuda::Stream stream;

    while (true) {
        cap >> frame;
        if (frame.empty())
            break;

        // Upload to GPU
        cv::cuda::GpuMat d_frame(frame);
        cv::cuda::GpuMat d_undistorted;

        // Undistort on GPU (using custom function since undistort isn't directly available in CUDA)
        cv::cuda::remap(d_frame, d_undistorted, d_mapx, d_mapy, cv::INTER_LINEAR, 0, cv::Scalar(), stream);

        // Run inference on the frame
        cv::Mat undistorted;
        d_undistorted.download(undistorted, stream);
        cv::Mat prediction_mask = inferencer.makePrediction(undistorted);

        // Upload prediction mask to GPU for faster processing
        cv::cuda::GpuMat d_prediction_mask(prediction_mask);
        cv::cuda::GpuMat d_visualization;

        // Convert to 8-bit on GPU
        d_prediction_mask.convertTo(d_visualization, CV_8U, 255.0, 0, stream);

        // Download for color mapping (if not available in CUDA)
        cv::Mat visualization;
        d_visualization.download(visualization, stream);

        // Apply color map
        cv::Mat colorized_mask;
        cv::applyColorMap(visualization, colorized_mask, cv::COLORMAP_JET);

        // Upload for resizing
        cv::cuda::GpuMat d_colorized_mask(colorized_mask);
        cv::cuda::GpuMat d_resized_mask;

        // Resize on GPU
        cv::cuda::resize(d_colorized_mask, d_resized_mask,
                        cv::Size(frame.cols * scale_factor, frame.rows * scale_factor),
                        0, 0, cv::INTER_LINEAR, stream);

        // Display logic
        cv::Mat resized_mask;
        d_resized_mask.download(resized_mask, stream);

        if (show_original) {
            // Resize original on GPU too
            cv::cuda::GpuMat d_resized_frame;
            cv::cuda::resize(d_frame, d_resized_frame,
                            cv::Size(frame.cols * scale_factor, frame.rows * scale_factor),
                            0, 0, cv::INTER_LINEAR, stream);

            cv::Mat resized_frame;
            d_resized_frame.download(resized_frame, stream);

            // Make sure both images have the same type before concatenation
            if (resized_frame.type() != resized_mask.type()) {
                cv::cvtColor(resized_mask, resized_mask, cv::COLOR_BGR2BGRA);
                cv::cvtColor(resized_frame, resized_frame, cv::COLOR_BGR2BGRA);
            }

            // Create a combined view
            try {
                cv::Mat combined;
                cv::hconcat(resized_frame, resized_mask, combined);
                cv::imshow(window_name, combined);
            } catch (const cv::Exception& e) {
                std::cerr << "Warning: Could not concatenate images: " << e.what() << std::endl;
                cv::imshow(window_name + " - Original", resized_frame);
                cv::imshow(window_name + " - Prediction", resized_mask);
            }
        } else {
            // Show only the prediction mask
            cv::imshow(window_name, resized_mask);
        }

        if (cv::waitKey(1) == 27)
            break; // Exit on ESC key
    }
}

// Destructor
CameraStreamer::~CameraStreamer() {
    cap.release();
    cv::destroyAllWindows();
}
