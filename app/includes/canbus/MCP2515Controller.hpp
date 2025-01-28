#ifndef MCP2515CONTROLLER_HPP
#define MCP2515CONTROLLER_HPP

#include <QObject>
#include "CANMessageProcessor.hpp"
#include "MCP2515Configurator.hpp"
#include "SPIController.hpp"
#include <string>

class MCP2515Controller : public QObject
{
    Q_OBJECT
public:
    /**
     * Constructs an MCP2515Controller object and initializes the SPI interface and related components.
     *
     * This constructor initializes the SPI controller and the associated configurator and message
     * processor objects. It attempts to open the SPI device using the provided device path. If
     * opening the device fails, an exception is thrown. After the device is successfully opened,
     * the constructor sets up the message handlers to handle incoming CAN messages.
     *
     * @param spiDevice The path to the SPI device to initialize the SPIController with.
     *
     * @throws std::runtime_error If the SPI device cannot be opened, indicating a failure in
     *         device initialization.
     */
    explicit MCP2515Controller(const std::string &spiDevice);

    /**
     * Destroys the MCP2515Controller object and cleans up resources.
     *
     * This destructor closes the SPI device to release the associated resources. It ensures
     * proper cleanup of the SPI connection when the MCP2515Controller object is destroyed.
     */
    ~MCP2515Controller();

    /**
     * Initializes the MCP2515 chip and configures it for normal operation.
     *
     * This function resets the MCP2515 chip and configures various parameters, including baud rate,
     * transmit and receive buffers, filters, masks, and interrupts. It then sets the chip to normal mode.
     * If any step fails, an exception is thrown to indicate the failure.
     *
     * @return Returns true if the initialization is successful and the chip is set to normal mode.
     *
     * @throws std::runtime_error If any initialization step fails, such as resetting the chip or setting it
     *         to normal mode, indicating a failure in the initialization process.
     */
    bool init();

    /**
     * Continuously reads CAN messages from the MCP2515 chip and processes them.
     *
     * This function enters a loop that reads CAN messages from the MCP2515 chip and processes them
     * using the message processor. It will continue reading and processing messages until the
     * `stopReadingFlag` is set to true. If an error occurs while processing a message, an error
     * message is logged. The function sleeps for 10 milliseconds between each iteration to manage
     * processing time.
     *
     * @throws std::exception If an error occurs during message reading or processing,
     *         an exception will be caught and logged.
     */
    void processReading();

    /**
     * Stops the CAN message reading loop.
     *
     * This function sets the `stopReadingFlag` to true, which causes the reading loop in
     * `processReading` to exit, stopping the continuous reading and processing of CAN messages.
     */
    void stopReading();

signals:
    void speedUpdated(float newSpeed);
    void rpmUpdated(int newRpm);

private:
    SPIController spiController;
    MCP2515Configurator configurator;
    CANMessageProcessor messageProcessor;
    bool stopReadingFlag = false;

    /**
     * Sets up message handlers for processing specific CAN frame IDs.
     *
     * This function registers custom handlers for specific CAN frame IDs. Each handler processes
     * the corresponding CAN message data and emits signals for relevant values such as speed or RPM.
     * The handlers are registered with the message processor, which is responsible for processing
     * incoming messages and invoking the appropriate handler based on the frame ID.
     *
     * - Frame ID 0x100: Extracts a float representing speed (scaled by 10) and emits the `speedUpdated` signal.
     * - Frame ID 0x200: Extracts a 16-bit value representing RPM and emits the `rpmUpdated` signal.
     */
    void setupHandlers();
};

#endif // MCP2515CONTROLLER_HPP
