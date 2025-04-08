#include "CameraStreamer.hpp"

int main() {
	CameraStreamer camera(0.5);  // Instantiate with a scale factor
	camera.start();
	return 0;
}

/*to compile it
	g++ main.cpp CameraStreamer.cpp -o camera_streamer `pkg-config --cflags --libs opencv4`
*/
