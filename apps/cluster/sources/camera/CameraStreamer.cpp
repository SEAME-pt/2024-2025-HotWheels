#include "CameraStreamer.hpp"

// Constructor
CameraStreamer::CameraStreamer(double scale, const std::string& win_name)
	: scale_factor(scale), window_name(win_name) {

	std::string pipeline = "nvarguscamerasrc ! video/x-raw(memory:NVMM), width=1280, height=720, format=(string)NV12, framerate=30/1 ! nvvidconv ! video/x-raw, format=(string)BGRx ! videoconvert ! video/x-raw, format=(string)BGR ! appsink";
	cap.open(pipeline, cv::CAP_GSTREAMER);

	inference = new TensorRTInferencer("model.engine");

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

		//cv::Mat resized_frame;
		cv::Mat cameraMatrix, distCoeffs;
		cv::FileStorage fs("camera_calibration.yml", FileStorage::READ);
		fs["camera_matrix"] >> cameraMatrix;
		fs["distortion_coefficients"] >> distCoeffs;
		fs.release();

		cv::Mat undistorted;
		cv::undistort(frame, undistorted, cameraMatrix, distCoeffs);

		cv::Mat prediction;
		//cv::resize(frame, resized_frame, cv::Size(), scale_factor, scale_factor, cv::INTER_LINEAR);
		prediction = inference.makePrediction(undistorted);
		cv::imshow(window_name, prediction);

		if (cv::waitKey(1) == 27)
			break; // Exit on ESC key
	}
}

// Destructor
CameraStreamer::~CameraStreamer() {
	cap.release();
	cv::destroyAllWindows();
}
