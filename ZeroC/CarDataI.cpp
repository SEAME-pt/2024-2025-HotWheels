#include "CarDataI.hpp"

namespace Data {

CarDataI::CarDataI(QObject *parent) : QObject(parent) {}

// Destructor implementation
CarDataI::~CarDataI() {
    if (communicator) {
        communicator->shutdown();
        communicator->destroy();
    }
}

// setJoystickValue implementation
void CarDataI::setJoystickValue(bool newValue, const Ice::Current&) {
    std::lock_guard<std::mutex> lock(joystick_mutex);
    joystick_enable = newValue;
    std::cout << "Joystick value set to: " << joystick_enable << std::endl;
}

// getJoystickValue implementation
bool CarDataI::getJoystickValue(const Ice::Current&) {
    std::lock_guard<std::mutex> lock(joystick_mutex);
    return joystick_enable;
}

// setCarTemperatureValue implementation
void CarDataI::setCarTemperatureValue(double newValue, const Ice::Current&) {
    std::lock_guard<std::mutex> lock(temperature_mutex);
    car_temperature = newValue;
    std::cout << "Car temperature value set to: " << car_temperature << std::endl;
}

// getCarTemperatureValue implementation
double CarDataI::getCarTemperatureValue(const Ice::Current&) {
    std::lock_guard<std::mutex> lock(temperature_mutex);
    return car_temperature;
}

// runServer implementation
void CarDataI::runServer(int argc, char **argv) {
    // Set Ice properties for thread pool size
    Ice::PropertiesPtr properties = Ice::createProperties();
    properties->setProperty("Ice.ThreadPool.Server.Size", "10");      // Set min thread pool size to 10
    properties->setProperty("Ice.ThreadPool.Server.SizeMax", "20");   // Can expand to 20 threads

    // Initialize Ice with the custom properties using InitializationData
    Ice::InitializationData initData;
    initData.properties = properties;

    communicator = Ice::initialize(argc, argv, initData);
    Ice::ObjectAdapterPtr adapter = communicator->createObjectAdapterWithEndpoints(
        "CarDataAdapter", "tcp -h 127.0.0.1 -p 10000");

    adapter->add(this, Ice::stringToIdentity("carData"));
    adapter->activate();

    std::cout << "Server is running, press Ctrl+C to stop." << std::endl;
    communicator->waitForShutdown();
}

} // namespace Data
