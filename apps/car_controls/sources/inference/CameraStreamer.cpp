#include "CameraStreamer.hpp"

// Constructor: initializes camera capture, inference reference, and settings
CameraStreamer::CameraStreamer(std::shared_ptr<TensorRTInferencer> inferencer, double scale, const std::string& win_name, bool show_orig)
	: scale_factor(scale), window_name(win_name), show_original(show_orig), m_inferencer(std::move(inferencer)), m_publisherObject(nullptr), m_running(true) {

	// Start publisher to pass frames to the cluster
	m_publisherObject = new Publisher(5556);

	gst_init(nullptr, nullptr);  // Initialize GStreamer

	// Define GStreamer pipeline for CSI camera
	//std::string pipeline = "nvarguscamerasrc sensor-mode=4 ! video/x-raw(memory:NVMM), width=1280, height=720, format=(string)NV12, framerate=60/1 ! nvvidconv ! video/x-raw, format=(string)BGRx ! videoconvert ! video/x-raw, format=(string)BGR ! appsink";

/* 	std::string pipeline =
		"nvarguscamerasrc sensor-mode=4 ! "
		"video/x-raw(memory:NVMM), width=1280, height=720, format=NV12, framerate=30/1 ! "
		"nvvidconv ! "
		"video/x-raw(memory:NVMM), format=RGBA ! "
		"appsink drop=true sync=false";

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
		cv::Size(1280, 720),
		CV_32FC1, mapx, mapy
	);  // Compute undistortion mapping

	d_mapx.upload(mapx);  // Upload X map to GPU
	d_mapy.upload(mapy);  // Upload Y map to GPU
}

// Main loop: capture, undistort, predict, visualize and render frames
void CameraStreamer::start() {
	initUndistortMaps();

	// Initialize Argus camera provider
	UniqueObj<CameraProvider> cameraProvider(CameraProvider::create());
	ICameraProvider* iCameraProvider = interface_cast<ICameraProvider>(cameraProvider);
	if (!iCameraProvider) {
		std::cerr << "Failed to get ICameraProvider interface." << std::endl;
		return;
	}

	// Get available camera devices
	std::vector<CameraDevice*> cameraDevices;
	iCameraProvider->getCameraDevices(&cameraDevices);
	if (cameraDevices.empty()) {
		std::cerr << "No camera devices available." << std::endl;
		return;
	}

	// Create capture session
	UniqueObj<CaptureSession> captureSession(iCameraProvider->createCaptureSession(cameraDevices[0]));
	ICaptureSession* iCaptureSession = interface_cast<ICaptureSession>(captureSession);
	if (!iCaptureSession) {
		std::cerr << "Failed to create capture session." << std::endl;
		return;
	}

	// Configure output stream settings
	UniqueObj<OutputStreamSettings> streamSettings(iCaptureSession->createOutputStreamSettings(STREAM_TYPE_EGL));
	IEGLOutputStreamSettings* iEglStreamSettings = interface_cast<IEGLOutputStreamSettings>(streamSettings);
	if (!iEglStreamSettings) {
		std::cerr << "Failed to get EGL stream settings interface." << std::endl;
		return;
	}
	iEglStreamSettings->setPixelFormat(PIXEL_FMT_YCbCr_420_888);
	iEglStreamSettings->setResolution(Size2D<uint32_t>(1280, 720));
	iEglStreamSettings->setMode(EGL_STREAM_MODE_MAILBOX);

	// Create output stream
	UniqueObj<OutputStream> outputStream(iCaptureSession->createOutputStream(streamSettings.get()));
	if (!outputStream) {
		std::cerr << "Failed to create output stream." << std::endl;
		return;
	}

	// Create frame consumer
	UniqueObj<FrameConsumer> consumer(FrameConsumer::create(outputStream.get()));
	IFrameConsumer* iFrameConsumer = interface_cast<IFrameConsumer>(consumer);
	if (!iFrameConsumer) {
		std::cerr << "Failed to create frame consumer." << std::endl;
		return;
	}

	// Create and submit capture request
	UniqueObj<Request> request(iCaptureSession->createRequest());
	IRequest* iRequest = interface_cast<IRequest>(request);
	if (!iRequest) {
		std::cerr << "Failed to create capture request." << std::endl;
		return;
	}
	iRequest->enableOutputStream(outputStream.get());
	if (iCaptureSession->repeat(request.get()) != STATUS_OK) {
		std::cerr << "Failed to start capture session." << std::endl;
		return;
	}

	// Setup decoder — assume you are feeding H.264 stream or live CSI
	// For CSI directly, you may need NvArgusCameraCapture instead.
	// Here we assume dmabuf_fd is obtained from capture loop

	int frame_count = 0;
	auto start_time = std::chrono::high_resolution_clock::now();

	while (m_running) {
		UniqueObj<Frame> frame(iFrameConsumer->acquireFrame());
		IFrame* iFrame = interface_cast<IFrame>(frame);
		if (!iFrame) {
			std::cerr << "Failed to acquire frame." << std::endl;
			continue;
		}

		// Get the image from the frame
		Image* image = iFrame->getImage();
		EGLStream::IImage* iImage = interface_cast<EGLStream::IImage>(image);
		if (!iImage) {
			std::cerr << "Failed to get IImage interface." << std::endl;
			continue;
		}

		// Get the EGLImage
		EGLImageKHR eglImage = iImage->getEGLImage();
		if (eglImage == EGL_NO_IMAGE_KHR) {
			std::cerr << "Failed to get EGLImage." << std::endl;
			continue;
		}

		// Map EGLImage to CUDA
		CUgraphicsResource cudaResource;
		CUresult cuResult = cuGraphicsEGLRegisterImage(&cudaResource, eglImage, CU_GRAPHICS_MAP_RESOURCE_FLAGS_NONE);
		if (cuResult != CUDA_SUCCESS) {
			std::cerr << "Failed to register EGLImage with CUDA." << std::endl;
			continue;
		}

		cuResult = cuGraphicsMapResources(1, &cudaResource, 0);
		if (cuResult != CUDA_SUCCESS) {
			std::cerr << "Failed to map CUDA resources." << std::endl;
			cuGraphicsUnregisterResource(cudaResource);
			continue;
		}

		CUeglFrame eglFrame;
		cuResult = cuGraphicsResourceGetMappedEglFrame(&eglFrame, cudaResource, 0, 0);
		if (cuResult != CUDA_SUCCESS) {
			std::cerr << "Failed to get mapped EGL frame." << std::endl;
			cuGraphicsUnmapResources(1, &cudaResource, 0);
			cuGraphicsUnregisterResource(cudaResource);
			continue;
		}

		cv::cuda::GpuMat d_frame(eglFrame.heigh, eglFrame.width, CV_8UC4, eglFrame.frame.pPitch[0]);
		cv::cuda::GpuMat d_undistorted;
		cv::cuda::remap(d_frame, d_undistorted, d_mapx, d_mapy, cv::INTER_LINEAR, 0, cv::Scalar(), stream);

		cv::cuda::GpuMat d_prediction_mask = m_inferencer->makePrediction(d_undistorted);

		cv::cuda::GpuMat d_mask_u8;
		d_prediction_mask.convertTo(d_mask_u8, CV_8U, 255.0, 0, stream);

		cv::cuda::GpuMat d_visualization;
		d_prediction_mask.convertTo(d_visualization, CV_8U, 255.0, 0, stream);

		cv::cuda::GpuMat d_resized_mask;
		cv::cuda::resize(d_visualization, d_resized_mask,
						 cv::Size(input_width * scale_factor, input_height * scale_factor),
						 0, 0, cv::INTER_LINEAR, stream);
		stream.waitForCompletion();

		if (m_publisherObject) {
			m_publisherObject->publishFrame("inference_frame", d_resized_mask);
		}

		// Unmap and unregister CUDA resources
		cuGraphicsUnmapResources(1, &cudaResource, 0);
		cuGraphicsUnregisterResource(cudaResource);

		frame_count++;
		auto now = std::chrono::high_resolution_clock::now();
		auto elapsed = std::chrono::duration_cast<std::chrono::seconds>(now - start_time).count();

		if (elapsed >= 1) {
			std::cout << "Average FPS: " << frame_count / static_cast<double>(elapsed) << std::endl;
			start_time = now;
			frame_count = 0;
		}
	}

	// Stop the capture session
	iCaptureSession->stopRepeat();
	iCaptureSession->waitForIdle();
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
