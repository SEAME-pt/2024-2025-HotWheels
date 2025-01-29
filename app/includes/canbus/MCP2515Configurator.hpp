#ifndef MCP2515CONFIGURATOR_HPP
#define MCP2515CONFIGURATOR_HPP

#include "ISPIController.hpp"
#include <cstdint>
#include <vector>

class MCP2515Configurator {
public:
    /**
     * Constructs an MCP2515Configurator object and initializes it with the provided SPI controller.
     *
     * This constructor takes an SPIController reference and associates it with the MCP2515Configurator,
     * enabling the configurator to interact with the SPI hardware for configuring the MCP2515 controller.
     * The SPIController is passed by reference, ensuring that the same controller is used throughout
     * the configurator’s lifetime.
     *
     * @param spiController A reference to the SPIController object that will be used for communication
     *                      with the MCP2515 controller.
     */
    explicit MCP2515Configurator(ISPIController &spiController);
    ~MCP2515Configurator() = default;

    /**
     * Resets the MCP2515 chip and verifies that it has entered configuration mode.
     *
     * This function sends a reset command to the MCP2515 chip, then waits for a brief period
     * to allow the reset process to complete. Afterward, it reads the chip's status register
     * to ensure that the chip has entered the configuration mode, which is indicated by
     * a specific value in the status register.
     *
     * @return Returns true if the chip has been successfully reset and is in configuration mode,
     *         false otherwise.
     */
    bool resetChip();

    /**
     * Configures the baud rate settings for the MCP2515 chip.
     *
     * This function sets the appropriate values for the baud rate prescaler and phase segments
     * in the MCP2515 configuration registers (CNF1, CNF2, CNF3). These settings define the
     * baud rate and timing characteristics for CAN communication. The values written to the
     * registers are specific to a desired baud rate configuration.
     */
    void configureBaudRate();

    /**
     * Configures the transmit (TX) buffer of the MCP2515 chip.
     *
     * This function clears the control register for the TX buffer (TXB0CTRL), effectively
     * resetting the buffer's configuration to its default state. This step is part of
     * setting up the MCP2515 for CAN message transmission.
     *
     * - TXB0CTRL: Transmit Buffer Control Register, which is cleared to ensure the
     *              TX buffer is properly initialized before use.
     */
    void configureTXBuffer();

    /**
     * Configures the receive (RX) buffer of the MCP2515 chip.
     *
     * This function sets the RX buffer control register (RXB0CTRL) to enable rollover and
     * configures the buffer to receive all CAN messages. By setting these bits, the MCP2515
     * will continuously store received messages, allowing for easier message processing.
     *
     * - RXB0CTRL: Receive Buffer Control Register, which is configured to enable rollover
     *              and set the RX mode to receive all messages.
     */
    void configureRXBuffer();

    /**
     * Configures the filters and masks for the MCP2515 chip.
     *
     * This function sets the values for the first filter and mask registers to allow the
     * MCP2515 to filter incoming CAN messages based on specific criteria. The filter and mask
     * settings determine which messages are passed to the RX buffer for processing.
     *
     * - Filter 0: The filter register is set to 0xFF, configuring the chip to accept specific
     *             messages based on the filter settings.
     * - Mask 0: The mask register is set to 0xFF, defining the acceptance criteria for the messages.
     */
    void configureFiltersAndMasks();

    /**
     * Configures the interrupt settings for the MCP2515 chip.
     *
     * This function enables the receive interrupt by setting the appropriate bit in the
     * CANINTE register. With this configuration, the MCP2515 will generate an interrupt
     * whenever a message is received, allowing the system to handle incoming messages
     * efficiently.
     *
     * - CANINTE: Interrupt Enable Register, where the receive interrupt bit is set to enable
     *            interrupts for received messages.
     */
    void configureInterrupts();

    /**
     * Sets the operation mode of the MCP2515 chip.
     *
     * This function writes the specified mode to the CANCTRL register, configuring the MCP2515
     * to operate in the desired mode. The mode can be used to select different operation modes
     * such as Normal, Sleep, or Configuration mode, based on the value provided.
     *
     * - CANCTRL: Control Register, where the mode is written to configure the chip's operation.
     * @param mode The mode to set for the MCP2515, passed as a byte value corresponding to
     *             the desired operation mode.
     */
    void setMode(uint8_t mode);

    /**
     * Verifies if the MCP2515 is operating in the expected mode.
     *
     * This function reads the CANSTAT register and checks the mode bits to ensure that the
     * MCP2515 is in the expected operation mode. The mode is verified by comparing the
     * current mode with the provided expected mode.
     *
     * - CANSTAT: Status Register, where the current operation mode is read.
     *
     * @param expectedMode The expected mode value to compare against the current mode.
     * @return Returns true if the current mode matches the expected mode, false otherwise.
     */
    bool verifyMode(uint8_t expectedMode);

    /**
     * Reads a CAN message from the MCP2515 chip.
     *
     * This function checks if a CAN message is available in the RX buffer. If data is available,
     * it reads the frame ID and the message length, followed by the message data itself. The
     * received data is stored in a vector and returned. After reading the message, the interrupt
     * flag is cleared to allow further message processing.
     *
     * @param frameID A reference to a variable where the frame ID of the received message
     *                will be stored.
     * @return A vector containing the received CAN message data.
     */
	std::vector<uint8_t> readCANMessage(uint16_t& frameID);
  
    static constexpr uint8_t RESET_CMD = 0xC0;
    static constexpr uint8_t CANCTRL = 0x0F;
    static constexpr uint8_t CANSTAT = 0x0E;
    static constexpr uint8_t CNF1 = 0x2A;
    static constexpr uint8_t CNF2 = 0x29;
    static constexpr uint8_t CNF3 = 0x28;
    static constexpr uint8_t TXB0CTRL = 0x30;
    static constexpr uint8_t RXB0CTRL = 0x60;
    static constexpr uint8_t CANINTF = 0x2C;
    static constexpr uint8_t CANINTE = 0x2B;
    static constexpr uint8_t RXB0SIDH = 0x61;
    static constexpr uint8_t RXB0SIDL = 0x62;
  
private:
    ISPIController &spiController;

    /**
     * Writes a value to a specified register on the MCP2515 chip.
     *
     * This function sends a write command to the specified register address and writes the
     * given value to it via the SPI interface. It uses the SPIController to send the data
     * to the MCP2515 chip.
     *
     * @param address The register address where the value will be written.
     * @param value The value to write to the register.
     */
    void writeRegister(uint8_t address, uint8_t value);

    /**
     * Reads a value from a specified register on the MCP2515 chip.
     *
     * This function sends a read command to the specified register address and retrieves
     * the value stored in that register via the SPI interface. It uses the SPIController
     * to read the data from the MCP2515 chip.
     *
     * @param address The register address to read from.
     * @return The value stored in the specified register.
     */
    uint8_t readRegister(uint8_t address);

    /**
     * Sends a command to the MCP2515 chip via SPI.
     *
     * This function sends a single command byte to the MCP2515 chip using the SPI interface.
     * The command is transmitted via the `spiTransfer` function of the SPIController, and
     * no data is expected to be received in response.
     *
     * @param command The command byte to send to the MCP2515 chip.
     */
    void sendCommand(uint8_t command);

};

#endif // MCP2515CONFIGURATOR_HPP
