#include "CameraStreamer.hpp"

int main() {
	CameraStreamer camera(0.5);  // Instantiate with a scale factor
	camera.start();
	return 0;
}
