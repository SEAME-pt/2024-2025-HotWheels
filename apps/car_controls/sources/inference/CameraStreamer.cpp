#include "CameraStreamer.hpp"

// Constructor: initializes camera capture, inference reference, and settings
CameraStreamer::CameraStreamer(std::shared_ptr<TensorRTInferencer> inferencer, double scale, const std::string& win_name, bool show_orig)
	: scale_factor(scale), window_name(win_name), show_original(show_orig), m_inferencer(std::move(inferencer)), m_publisherObject(nullptr), m_running(true) {

	// Start publisher to pass frames to the cluster
	m_publisherObject = new Publisher(5556);

/* 	// Define GStreamer pipeline for CSI camera
	std::string pipeline = "nvarguscamerasrc sensor-mode=4 ! video/x-raw(memory:NVMM), width=1280, height=720, format=(string)NV12, framerate=60/1 ! nvvidconv ! video/x-raw, format=(string)BGRx ! videoconvert ! video/x-raw, format=(string)BGR ! appsink";

	cap.open(pipeline, cv::CAP_GSTREAMER); // Open camera stream with GStreamer

	if (!cap.isOpened()) {  // Check if camera opened successfully
		std::cerr << "Error: Could not open CSI camera" << std::endl;
		exit(-1);  // Terminate if failed
	} */
}

// Destructor: clean up resources
CameraStreamer::~CameraStreamer() {
	delete m_publisherObject;
	m_publisherObject = nullptr;

	stop();  // Stop the camera stream

	if (cap.isOpened()) {
		cap.release(); // Release camera
	}

	cudaDeviceSynchronize();  // Ensure all CUDA operations are complete

	if (cuda_resource) {
		cudaGraphicsUnregisterResource(cuda_resource);  // Unregister CUDA graphics resource
		cuda_resource = nullptr;
	}

	if (window)
	{
		std::cout << "[~CameraStreamer] Destroying window..." << std::endl;
		glfwMakeContextCurrent(window);  // Ensure valid context
		glDeleteTextures(1, &textureID); // Clean texture
		glfwDestroyWindow(window);       // Destroy window
		std::cout << "[~CameraStreamer] Terminating GLFW..." << std::endl;
		glfwTerminate();
	}
	std::cout << "[~CameraStreamer] Destructor done." << std::endl;
}

// Initialize OpenGL context and prepare a texture for CUDA interop
void CameraStreamer::initOpenGL() {
	if (!glfwInit()) {  // Initialize GLFW library
		std::cerr << "Failed to initialize GLFW!" << std::endl;
		exit(-1);
	}

	window_width = static_cast<int>(1280 * scale_factor);  // Calculate scaled window width
	window_height = static_cast<int>(720 * scale_factor);  // Calculate scaled window height

	window = glfwCreateWindow(window_width, window_height, window_name.c_str(), NULL, NULL);  // Create OpenGL window
	if (!window) {  // Check if window creation failed
		std::cerr << "Failed to create GLFW window!" << std::endl;
		glfwTerminate();
		exit(-1);
	}

	glfwMakeContextCurrent(window);  // Make window's OpenGL context current
	glewInit();  // Initialize GLEW (needed to manage OpenGL extensions)

	glGenTextures(1, &textureID);  // Generate OpenGL texture ID
	glBindTexture(GL_TEXTURE_2D, textureID);  // Bind the texture
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);  // Set texture minification filter
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);  // Set texture magnification filter

	glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA8, window_width, window_height, 0, GL_RGBA, GL_UNSIGNED_BYTE, nullptr);  // Allocate empty texture (RGBA8 format)

	// Register the OpenGL texture with CUDA
	cudaError_t err = cudaGraphicsGLRegisterImage(&cuda_resource, textureID, GL_TEXTURE_2D, cudaGraphicsRegisterFlagsWriteDiscard);
	if (err != cudaSuccess) {  // Check CUDA-OpenGL interop registration
		std::cerr << "Failed to register OpenGL texture with CUDA: " << cudaGetErrorString(err) << std::endl;
		exit(-1);
	}
}

// Upload a GpuMat frame (on GPU) directly into an OpenGL texture using CUDA interop
void CameraStreamer::uploadFrameToTexture(const cv::cuda::GpuMat& gpuFrame) {
	cv::cuda::GpuMat d_rgba_frame;
	if (gpuFrame.channels() == 3) {  // If input is BGR
		cv::cuda::cvtColor(gpuFrame, d_rgba_frame, cv::COLOR_BGR2RGBA);  // Convert BGR to RGBA
	} else if (gpuFrame.channels() == 1) {  // If grayscale
		cv::cuda::cvtColor(gpuFrame, d_rgba_frame, cv::COLOR_GRAY2RGBA);  // Convert grayscale to RGBA
	} else {
		d_rgba_frame = gpuFrame;  // Already RGBA
	}

	cudaGraphicsMapResources(1, &cuda_resource, 0);  // Map OpenGL texture for CUDA access

	cudaArray_t texture_ptr;
	cudaGraphicsSubResourceGetMappedArray(&texture_ptr, cuda_resource, 0, 0);  // Get CUDA array pointer linked to OpenGL texture

	cudaMemcpy2DToArray(
		texture_ptr,
		0, 0,
		d_rgba_frame.ptr(),      // Source pointer (GPU memory)
		d_rgba_frame.step,       // Source stride
		d_rgba_frame.cols * d_rgba_frame.elemSize(), // Width in bytes
		d_rgba_frame.rows,       // Height in rows
		cudaMemcpyDeviceToDevice // GPU-to-GPU memory copy
	);

	cudaGraphicsUnmapResources(1, &cuda_resource, 0);  // Unmap resource after copy
}

// Render the current OpenGL texture to the screen
void CameraStreamer::renderTexture() {
	glClear(GL_COLOR_BUFFER_BIT);  // Clear the screen
	glEnable(GL_TEXTURE_2D);  // Enable 2D texturing
	glBindTexture(GL_TEXTURE_2D, textureID);  // Bind the texture to use

	glBegin(GL_QUADS);  // Start drawing a rectangle
	glTexCoord2f(0, 1); glVertex2f(-1.0f, -1.0f);  // Bottom-left
	glTexCoord2f(1, 1); glVertex2f(1.0f, -1.0f);   // Bottom-right
	glTexCoord2f(1, 0); glVertex2f(1.0f, 1.0f);    // Top-right
	glTexCoord2f(0, 0); glVertex2f(-1.0f, 1.0f);   // Top-left
	glEnd();  // End drawing rectangle

	glfwSwapBuffers(window);  // Swap front and back buffers (double buffering)
	glfwPollEvents();  // Process window events
}

// Load camera calibration file and initialize undistortion maps (upload to GPU)
void CameraStreamer::initUndistortMaps() {
	cv::Mat cameraMatrix, distCoeffs;
	cv::FileStorage fs("/home/hotweels/apps/camera_calibration.yml", cv::FileStorage::READ);  // Open calibration file

	if (!fs.isOpened()) {
		std::cerr << "[Error] Failed to open camera_calibration.yml" << std::endl;
		return;  // Handle file opening error
	}

	fs["camera_matrix"] >> cameraMatrix;  // Read camera matrix
	fs["distortion_coefficients"] >> distCoeffs;  // Read distortion coefficients
	fs.release();  // Close file

	cv::Mat mapx, mapy;
	cv::initUndistortRectifyMap(
		cameraMatrix, distCoeffs, cv::Mat(), cameraMatrix,
		cv::Size(cap.get(cv::CAP_PROP_FRAME_WIDTH), cap.get(cv::CAP_PROP_FRAME_HEIGHT)),
		CV_32FC1, mapx, mapy
	);  // Compute undistortion mapping

	d_mapx.upload(mapx);  // Upload X map to GPU
	d_mapy.upload(mapy);  // Upload Y map to GPU
}

// Main loop: capture, undistort, predict, visualize and render frames
void CameraStreamer::start() {
	initUndistortMaps();  // Initialize camera undistortion maps
	//initOpenGL();  // Initialize OpenGL and CUDA interop

	const char* device = "/dev/video0";
	int cam_fd = open(device, O_RDWR);
	if (cam_fd < 0) {
		perror("Failed to open camera");
		return;
	}

	std::cout << "Camera opened successfully." << std::endl;

	// Request buffer
	v4l2_requestbuffers req = {};
	req.count = 1;
	req.type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
	req.memory = V4L2_MEMORY_MMAP;
	if (ioctl(cam_fd, VIDIOC_REQBUFS, &req) == -1) {
		perror("VIDIOC_REQBUFS");
		close(cam_fd);
		return;
	}

	std::cout << "Buffer requested." << std::endl;

	// Query buffer
	v4l2_buffer buf = {};
	buf.type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
	buf.memory = V4L2_MEMORY_MMAP;
	buf.index = 0;
	if (ioctl(cam_fd, VIDIOC_QUERYBUF, &buf) == -1) {
		perror("VIDIOC_QUERYBUF");
		close(cam_fd);
		return;
	}

	std::cout << "Buffer queried." << std::endl;

	void* buffer = mmap(NULL, buf.length, PROT_READ | PROT_WRITE, MAP_SHARED, cam_fd, buf.m.offset);
	if (buffer == MAP_FAILED) {
		perror("mmap");
		close(cam_fd);
		return;
	}

	std::cout << "Buffer memory mapped." << std::endl;

	// Start streaming
	v4l2_buf_type type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
	if (ioctl(cam_fd, VIDIOC_STREAMON, &type) == -1) {
		perror("VIDIOC_STREAMON");
		munmap(buffer, buf.length);
		close(cam_fd);
		return;
	}

	std::cout << "Streaming started." << std::endl;

	cv::cuda::Stream stream;
	size_t width = 1280, height = 720;
	size_t size = width * height * 2; // YUYV = 2 bytes per pixel
	unsigned char* d_yuyv;
	cudaMalloc(&d_yuyv, size);

	auto start_time = std::chrono::high_resolution_clock::now();
	int frame_count = 0;

	//while (!glfwWindowShouldClose(window)) {  // Main loop until window closed
	while (m_running) {  // Main loop until stop signal
		std::cout << "Waiting for frame..." << std::endl;
		if (ioctl(cam_fd, VIDIOC_QBUF, &buf) == -1) {
			perror("VIDIOC_QBUF");
			break;
		}

		std::cout << "Buffer queued." << std::endl;

		if (ioctl(cam_fd, VIDIOC_DQBUF, &buf) == -1) {
			perror("VIDIOC_DQBUF");
			break;
		}

		std::cout << "Buffer dequeued." << std::endl;

		// Copy YUYV data to GPU
		cudaMemcpy(d_yuyv, buffer, size, cudaMemcpyHostToDevice);

		std::cout << "YUYV data copied to GPU." << std::endl;

		// Convert YUYV to BGR using OpenCV CUDA
		cv::cuda::GpuMat yuyv(height, width, CV_8UC2, d_yuyv);
		cv::cuda::GpuMat bgr;
		cv::cuda::cvtColor(yuyv, bgr, cv::COLOR_YUV2BGR_YUY2, 0, stream);

		cv::cuda::GpuMat d_undistorted;
		cv::cuda::remap(bgr, d_undistorted, d_mapx, d_mapy, cv::INTER_LINEAR, 0, cv::Scalar(), stream);  // Undistort frame

		cv::cuda::GpuMat d_prediction_mask = m_inferencer->makePrediction(d_undistorted);  // Run model inference

		// Convert to 8-bit (0 or 255) in a new GpuMat
		cv::cuda::GpuMat d_mask_u8;
		d_prediction_mask.convertTo(d_mask_u8, CV_8U, 255.0);  // Multiply 0/1 float to 0/255

		cv::Mat binary_mask_cpu;
		d_mask_u8.download(binary_mask_cpu, stream);
		cv::threshold(binary_mask_cpu, binary_mask_cpu, 128, 255, cv::THRESH_BINARY);
		stream.waitForCompletion();  // Ensure async operations are complete

		// Convert model output to 8-bit binary mask on GPU
		cv::cuda::GpuMat d_visualization;
		d_prediction_mask.convertTo(d_visualization, CV_8U, 255.0, 0, stream);

		cv::cuda::GpuMat d_resized_mask;

		cv::cuda::resize(d_visualization, d_resized_mask,
						 cv::Size(width * scale_factor, height * scale_factor),
						 0, 0, cv::INTER_LINEAR, stream);  // Resize for display
		stream.waitForCompletion();  // Synchronize

		if (m_publisherObject) {
			m_publisherObject->publishFrame("inference_frame", d_resized_mask);  // Publish the frame
		}

		frame_count++;
		auto now = std::chrono::high_resolution_clock::now();
		auto elapsed = std::chrono::duration_cast<std::chrono::seconds>(now - start_time).count();

		if (elapsed >= 1) {
			std::cout << "Average FPS: " << frame_count / static_cast<double>(elapsed) << std::endl;
			start_time = now;
			frame_count = 0;
		}

		//uploadFrameToTexture(d_resized_mask);  // Upload final result to OpenGL
		//renderTexture();  // Render it
	}

	cudaFree(d_yuyv);
	munmap(buffer, buf.length);
	close(cam_fd);
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
