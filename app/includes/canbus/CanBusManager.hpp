#ifndef CANBUSMANAGER_HPP
#define CANBUSMANAGER_HPP

#include <QObject>
#include <QThread>
#include "IMCP2515Controller.hpp"

class CanBusManager : public QObject
{
    Q_OBJECT
public:
    /**
     * Initializes the CanBusManager object by setting up the MCP2515Controller
     * and connecting relevant signals and slots.
     *
     * @param spi_device A string representing the SPI device to initialize
     *                   the MCP2515Controller with.
     * @param parent A pointer to the parent QObject, typically used for object
     *               hierarchy and memory management.
     *
     * This constructor creates a new MCP2515Controller instance using the provided
     * SPI device and sets up connections between the controller and the CanBusManager
     * to handle updates related to speed and RPM.
     */
    explicit CanBusManager(const std::string &spi_device, QObject *parent = nullptr);
    CanBusManager(IMCP2515Controller *controller, QObject *parent = nullptr);

    /**
     * Destructor for the CanBusManager object.
     *
     * This destructor stops the reading process in the MCP2515Controller, disconnects
     * the thread, and waits for the thread to finish before deleting the controller
     * and thread objects.
     */
    ~CanBusManager();

    /**
     * Initializes the CanBusManager by setting up and starting the MCP2515Controller
     * in a separate thread.
     *
     * This function first attempts to initialize the MCP2515Controller. If initialization
     * fails, it logs an error and returns false. If initialization is successful, it
     * creates a new QThread for the controller, moves the controller to the new thread,
     * and establishes necessary signal-slot connections to handle the reading process.
     * The thread is started, allowing the controller to begin processing data in the background.
     *
     * @return Returns true if initialization and thread setup are successful, false
     *         otherwise.
     */
    bool initialize();

signals:
    /**
     * Emits the `speedUpdated` signal with the updated speed value.
     *
     * This function is called when the speed value has been updated. It activates the `speedUpdated` signal
     * with the updated speed value (`_t1`). The value is passed to the signal using the `QMetaObject::activate`
     * method, which ensures that the connected slots are properly notified with the updated speed.
     *
     * @param _t1 The updated speed value to be emitted via the `speedUpdated` signal.
     */
    void speedUpdated(float newSpeed);

    /**
     * Emits the `rpmUpdated` signal with the updated RPM value.
     *
     * This function is called when the RPM value has been updated. It activates the `rpmUpdated` signal
     * with the updated RPM value (`_t1`). The value is passed to the signal using the `QMetaObject::activate`
     * method, ensuring that the connected slots are notified with the updated RPM.
     *
     * @param _t1 The updated RPM value to be emitted via the `rpmUpdated` signal.
     */
    void rpmUpdated(int newRpm);

private:
    IMCP2515Controller *m_controller = nullptr;
    QThread *m_thread = nullptr;
    bool ownsMCP2515Controller = false;

    void connectSignals();
};

#endif // CANBUSMANAGER_HPP
