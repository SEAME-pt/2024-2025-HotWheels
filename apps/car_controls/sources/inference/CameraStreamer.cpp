#include "../../includes/inference/CameraStreamer.hpp"


// Constructor: initializes camera capture, inference reference, and settings
CameraStreamer::CameraStreamer(double scale)
	: scale_factor(scale), m_publisherFrameObject(nullptr), m_running(true) {

	// Start publisher to pass frames to lane detection and object detection
	m_publisherFrameObject = new Publisher(5557);

	segmentationInferencer = std::make_shared<TensorRTInferencer>("/home/hotweels/dev/model_loader/models/model.engine");
	yoloInferencer = std::make_shared<YOLOv5TRT>("/home/hotweels/cam_calib/models/yolov5m_updated.engine", "/home/hotweels/cam_calib/models/labels.txt");

	// Define GStreamer pipeline for CSI camera
	std::string pipeline = "nvarguscamerasrc sensor-mode=4 ! "
			"video/x-raw(memory:NVMM), width=1280, height=720, "
			"format=(string)NV12, framerate=30/1 ! "
			"nvvidconv ! video/x-raw, format=(string)BGRx ! "
			"videoconvert ! video/x-raw, format=(string)BGR ! "
			"appsink drop=1 buffers=1";

	std::cout << "[CameraStreamer] Using GStreamer pipeline: " << pipeline << std::endl;

	cap.open(pipeline, cv::CAP_GSTREAMER); // Open camera stream with GStreamer

	std::cout << "[CameraStreamer] Camera opened." << std::endl;

	if (!cap.isOpened()) {  // Check if camera opened successfully
		std::cerr << "Error: Could not open CSI camera" << std::endl;
		exit(-1);  // Terminate if failed
	}
}

// Destructor: clean up resources
CameraStreamer::~CameraStreamer() {
	stop();  // Stop the camera stream

	// Join all threads safely
	if (captureThread.joinable()) captureThread.join();
	if (segmentationThread.joinable()) segmentationThread.join();
	if (detectionThread.joinable()) detectionThread.join();

	if (cap.isOpened()) {
		cap.release(); // Release camera
	}

	cudaDeviceSynchronize();  // Ensure all CUDA operations are complete

	if (cuda_resource) {
		cudaGraphicsUnregisterResource(cuda_resource);  // Unregister CUDA graphics resource
		cuda_resource = nullptr;
	}

	delete m_publisherFrameObject;
	m_publisherFrameObject = nullptr;

	std::cout << "[~CameraStreamer] Destructor done." << std::endl;
}

void CameraStreamer::segmentationWorker() {
	while (m_running) {
		cv::Mat frame;
		if (segmentationBuffer.getFrame(frame)) {
			auto start = std::chrono::high_resolution_clock::now();

			segmentationInferencer->doInference(frame);

			auto end = std::chrono::high_resolution_clock::now();
			auto duration_ms = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();

			std::cout << "[Segmentation] Inference time: " << duration_ms << " ms" << std::endl;
		} else {
			std::this_thread::sleep_for(std::chrono::milliseconds(1));
		}
	}
}

void CameraStreamer::detectionWorker() {
	while (m_running) {
		cv::Mat frame;
		if (detectionBuffer.getFrame(frame)) {
			yoloInferencer->process_image(frame);
		} else {
			std::this_thread::sleep_for(std::chrono::milliseconds(1));
		}
	}
}

// Main loop: capture, undistort, predict, visualize and render frames
void CameraStreamer::start() {
	m_running = true;

	captureThread = std::thread(&CameraStreamer::captureLoop, this);
	segmentationThread = std::thread(&CameraStreamer::segmentationWorker, this);
	detectionThread = std::thread(&CameraStreamer::detectionWorker, this);

	captureThread.join();
	segmentationThread.join();
	detectionThread.join();
}

void CameraStreamer::captureLoop() {
	const int target_fps = 15;
	const int frame_interval_ms = 1000 / target_fps;

	while (m_running) {
		auto start_time = std::chrono::steady_clock::now();

		cv::Mat frame;
		cap >> frame;

		if (frame.empty()) {
			std::cerr << "Empty frame, exiting" << std::endl;
			break;
		}

		segmentationBuffer.update(frame);
		detectionBuffer.update(frame);

		auto end_time = std::chrono::steady_clock::now();
		auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time).count();
		if (elapsed < frame_interval_ms) {
			std::this_thread::sleep_for(std::chrono::milliseconds(frame_interval_ms - elapsed));
		}
	}
}

void CameraStreamer::stop() {
	if (!m_running) return;
	m_running = false;

	// Wait for any CUDA operations to finish
	try {
		cudaDeviceSynchronize();
	} catch (const std::exception& e) {
		std::cerr << "CUDA sync error in stop(): " << e.what() << std::endl;
	}
	std::this_thread::sleep_for(std::chrono::milliseconds(100));

	std::cout << "[CameraStreamer] Shutdown complete." << std::endl;
}
