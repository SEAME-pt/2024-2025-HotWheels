/*!
 * @file ClientThread.cpp
 * @brief Implementation of the ClientThread class.
 * @version 0.1
 * @date 2025-02-12
 * @details This file contains the implementation of the ClientThread class,
 * which is responsible for handling the client-side communication with the server.
 * @author Félix LE BIHAN (@Fle-bihh)
 * @author Tiago Pereira (@t-pereira06)
 * @author Ricardo Melo (@reomelo)
 * @author Michel Batista (@MicchelFAB)
 *
 * @copyright Copyright (c) 2025
 */

#include "ClientThread.hpp"

/**
 * @brief Construct a new Client Thread:: Client Thread object
 * 
 * @param parent 
 */
ClientThread::ClientThread(QObject *parent) : QObject(parent) {}


/**
 * @brief Destructor for the ClientThread class.
 * @details Shuts down and destroys the Ice communicator if it is initialized,
 * ensuring a clean termination of client-server communication.
 */
ClientThread::~ClientThread() {
     if (communicator) {
        communicator->shutdown();
        communicator->destroy();
    }
}

/**
 * @brief Run the client by initializing the Ice communicator and connecting to the server.
 * @param argc The number of command-line arguments.
 * @param argv The array of command-line arguments.
 * @details This method initializes the Ice communicator using the given argc and argv and
 * connects to the server by specifying the proxy endpoint. The carData proxy is then cast
 * to the correct type (carDataPrx). The connected flag is set to true once the connection
 * is established and the main thread is notified via a condition variable. The client then
 * enters a loop where it waits for requests and processes them. The loop is exited when the
 * running flag is set to false and the communicator is gracefully shutdown.
 */
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

/**
 * @brief Set the joystick value on the server.
 * @param value The new value to set the joystick to.
 * @details This method sets the joystick value on the server by calling the
 * setJoystickValue method on the carData proxy. It first waits until the client
 * is connected to the server and then checks if the carData proxy is valid.
 * If the proxy is valid, it calls the method, otherwise it prints an error
 * message.
 */
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

/**
 * @brief Get the joystick value from the server.
 * @details This method gets the joystick value from the server by calling the
 * getJoystickValue method on the carData proxy. It first waits until the client
 * is connected to the server and then checks if the carData proxy is valid.
 * If the proxy is valid, it calls the method and returns the joystick value,
 * otherwise it returns false and prints an error message.
 * @return The current value of the joystick.
 */
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

/**
 * @brief Sets the running flag of the client thread.
 * @details This method sets the running flag of the client thread, which
 * controls whether the client is connected to the server or not. This flag is
 * used to stop the client thread gracefully when requested.
 * @param value The new value for the running flag.
 */
void ClientThread::setRunning(bool value) {
    std::lock_guard<std::mutex> lock(mtx);
    this->running = value;
}
