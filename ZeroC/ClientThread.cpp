#include "ClientThread.hpp"

ClientThread::ClientThread(QObject *parent) : QObject(parent) {}

// Destructor stops the client thread gracefully
ClientThread::~ClientThread() {
     if (communicator) {
        communicator->shutdown();
        communicator->destroy();
    }
}

// The client thread runs this method
void ClientThread::runClient(int argc, char* argv[]) {
    try {
        // Initialize Ice communicator
        communicator = Ice::initialize(argc, argv);

        // Connect to the server by specifying the proxy endpoint
        base = communicator->stringToProxy("carData:tcp -h 127.0.0.1 -p 10000");

        // Cast the base proxy to the correct type (carDataPrx)
        carData = Data::CarDataPrx::checkedCast(base);
        if (!carData) {
            throw std::runtime_error("Invalid proxy, failed to cast.");
        }

        // Set the connected flag to true once the connection is established
        {
            std::lock_guard<std::mutex> lock(mtx);
            connected = true;
        }

        // Notify the main thread that the client is connected
        cv.notify_all();

        // Keep the client running and processing requests
        while (running) {
            std::this_thread::sleep_for(std::chrono::milliseconds(100));
        }

        // Gracefully shutdown the communicator
        communicator->destroy();
    } catch (const Ice::Exception& e) {
        std::cerr << "Ice Exception: " << e.what() << std::endl;
    }
}

// Methods to interact with the server via the client
void ClientThread::setJoystickValue(bool value) {
    // Wait until the client is connected to the server
    std::unique_lock<std::mutex> lock(mtx);
    cv.wait(lock, [this] { return connected; });

    // Ensure the carData proxy is valid before making the request
    if (carData) {
        try {
            carData->setJoystickValue(value);
        } catch (const Ice::Exception& e) {
            std::cerr << "Error setting joystick value: " << e.what() << std::endl;
        }
    } else {
        std::cerr << "Joystick proxy is not valid!" << std::endl;
    }
}

bool ClientThread::getJoystickValue() {
    // Wait until the client is connected to the server
    std::unique_lock<std::mutex> lock(mtx);
    cv.wait(lock, [this] { return connected; });

    // Ensure the carData proxy is valid before making the request
    if (carData) {
        try {
            bool state = carData->getJoystickValue();
            return state;
        } catch (const Ice::Exception& e) {
            std::cerr << "Error getting joystick value: " << e.what() << std::endl;
            return false;  // Return default value on error
        }
    } else {
        std::cerr << "Joystick proxy is not valid!" << std::endl;
        return false;  // Return default value if the proxy is invalid
    }
}

void ClientThread::setRunning(bool value) {
    std::lock_guard<std::mutex> lock(mtx);
    this->running = value;
}
