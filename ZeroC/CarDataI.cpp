/*!
 * @file CarDataI.cpp
 * @brief Implementation of the CarDataI class
 * @version 0.1
 * @date 2025-02-12
 * @details This file contains the implementation of the CarDataI class, which
 * is responsible for handling the car data.
 * @author Félix LE BIHAN (@Fle-bihh)
 * @author Tiago Pereira (@t-pereira06)
 * @author Ricardo Melo (@reomelo)
 * @author Michel Batista (@MicchelFAB)
 *
 * @copyright Copyright (c) 2025
 */

#include "CarDataI.hpp"

/**
 * @brief Namespace for the car data
 * @namespace Data
 */
namespace Data
{

    /**
     * @brief Constructor for the CarDataI class
     * @param parent The parent QObject
     */
    CarDataI::CarDataI(QObject *parent) : QObject(parent) {}

    /**
     * @brief Destructor for the CarDataI class
     * @details This destructor is responsible for stopping the Ice communicator
     * and destroying it.
     */
    CarDataI::~CarDataI()
    {
        if (communicator)
        {
            communicator->shutdown();
            communicator->destroy();
        }
    }

    /**
     * @brief Set the joystick value
     * @details This method is responsible for setting the joystick value.
     * @param newValue The new value for the joystick
     * @param current The current Ice context
     */
    void CarDataI::setJoystickValue(bool newValue, const Ice::Current &)
    {
        std::lock_guard<std::mutex> lock(joystick_mutex);
        joystick_enable = newValue;
        std::cout << "Joystick value set to: " << joystick_enable << std::endl;
    }

    /**
     * @brief Get the joystick value
     * @details This method is responsible for getting the joystick value.
     * @return The current value of the joystick
     */
    bool CarDataI::getJoystickValue(const Ice::Current &)
    {
        std::lock_guard<std::mutex> lock(joystick_mutex);
        return joystick_enable;
    }

    /**
     * @brief Set the car temperature value
     * @details This method sets the temperature of the car to the specified value.
     *          It is thread-safe and uses a mutex to ensure that the temperature
     *          is updated without race conditions. Logs the new temperature value
     *          to the console.
     * @param newValue The new temperature value to be set
     * @param current The current Ice context
     */
    void CarDataI::setCarTemperatureValue(double newValue, const Ice::Current &)
    {
        std::lock_guard<std::mutex> lock(temperature_mutex);
        car_temperature = newValue;
        std::cout << "Car temperature value set to: " << car_temperature << std::endl;
    }

    /**
     * @brief Get the car temperature value
     * @details This method is responsible for getting the car temperature value.
     *          It is thread-safe and uses a mutex to ensure that the temperature
     *          is retrieved without race conditions.
     * @return The current value of the car temperature
     */
    double CarDataI::getCarTemperatureValue(const Ice::Current &)
    {
        std::lock_guard<std::mutex> lock(temperature_mutex);
        return car_temperature;
    }

    /**
     * @brief Runs the Ice server.
     * @details This method runs the Ice server by initializing the
     * Ice communicator with the given arguments and custom properties.
     * The custom properties set the minimum and maximum thread pool size
     * to 10 and 20 respectively. The server is then activated and will
     * listen on the specified endpoint ("tcp -h 127.0.0.1 -p 10000").
     * The server will run until it is stopped with Ctrl+C.
     * @param argc The number of command-line arguments.
     * @param argv The array of command-line arguments.
     */
    void CarDataI::runServer(int argc, char **argv)
    {
        // Set Ice properties for thread pool size
        Ice::PropertiesPtr properties = Ice::createProperties();
        properties->setProperty("Ice.ThreadPool.Server.Size", "10");    // Set min thread pool size to 10
        properties->setProperty("Ice.ThreadPool.Server.SizeMax", "20"); // Can expand to 20 threads

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
