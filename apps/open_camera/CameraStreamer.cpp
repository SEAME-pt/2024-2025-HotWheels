#include "CameraStreamer.hpp"

// Constructor
CameraStreamer::CameraStreamer(double scale, const std::string& win_name)
	: scale_factor(scale), window_name(win_name) {

	std::string pipeline = "nvarguscamerasrc ! video/x-raw(memory:NVMM), width=1280, height=720, format=(string)NV12, framerate=30/1 ! nvvidconv ! video/x-raw, format=(string)BGRx ! videoconvert ! video/x-raw, format=(string)BGR ! appsink";
	cap.open(pipeline, cv::CAP_GSTREAMER);

	inferencer = new TensorRTInferencer("model.engine");

	if (!cap.isOpened()) {
		std::cerr << "Error: Could not open CSI camera" << std::endl;
		exit(-1);
	}
}

// Start streaming
void CameraStreamer::start() {
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
}

// Destructor
CameraStreamer::~CameraStreamer() {
	cap.release();
	cv::destroyAllWindows();
}
