#ifndef SPICONTROLLER_HPP
#define SPICONTROLLER_HPP

#include <cstdint>
#include <string>

class SPIController
{
public:
    enum class Opcode : uint8_t { Write = 0x02, Read = 0x03 };

    /**
     * Constructs an SPIController object and initializes default settings.
     *
     * This constructor initializes the SPIController with default values for the SPI file descriptor,
     * mode, bits per word, and speed. The `spi_fd` is set to -1 to indicate that the device is not
     * yet opened. Default settings are used for the mode, bits per word, and speed until they are
     * explicitly configured.
     */
    SPIController();

    /**
     * Destroys the SPIController object and cleans up resources.
     *
     * This destructor calls the `closeDevice` function to ensure that the SPI device is properly closed
     * and any resources associated with the SPI connection are released when the SPIController object is destroyed.
     */
    ~SPIController();

    /**
     * Opens the specified SPI device for communication.
     *
     * This function attempts to open the SPI device specified by the `device` string. If the device
     * cannot be opened, it throws an exception with an error message. If the device is successfully
     * opened, it returns true. The `spi_fd` file descriptor is set to the opened device's file descriptor.
     *
     * @param device The path to the SPI device to open.
     * @return Returns true if the device is successfully opened.
     *
     * @throws std::runtime_error If the device cannot be opened, indicating a failure in the SPI device
     *         initialization.
     */
    bool openDevice(const std::string &device);

    /**
     * Configures the SPI interface with the specified settings.
     *
     * This function configures the SPI device with the provided mode, bits per word, and speed.
     * It checks if the device is open and throws an exception if the device is not open. The function
     * then applies the configuration settings using the appropriate `ioctl` system calls. If any configuration
     * step fails, an exception is thrown with an error message.
     *
     * @param mode The SPI mode to set for the communication.
     * @param bits The number of bits per word for SPI communication.
     * @param speed The SPI clock speed (in Hz).
     *
     * @throws std::runtime_error If the device is not open or if any configuration step fails
     *         (setting the mode, bits per word, or speed).
     */
    void configure(uint8_t mode, uint8_t bits, uint32_t speed);

    /**
     * Writes a byte of data to the specified address on the SPI device.
     *
     * This function sends a write command to the SPI device, using the specified address and data byte.
     * The command is formed by combining the write opcode, the address, and the data byte. The SPI transfer
     * is then performed using the `spiTransfer` function to send the data to the device.
     *
     * @param address The address to write the data to on the SPI device.
     * @param data The data byte to write to the specified address.
     */
    void writeByte(uint8_t address, uint8_t data);

    /**
     * Reads a byte of data from the specified address on the SPI device.
     *
     * This function sends a read command to the SPI device, using the specified address. The function
     * performs a SPI transfer, where the read command is sent along with the address. The data byte
     * returned from the SPI device is stored in the response buffer and returned by the function.
     *
     * @param address The address to read the data from on the SPI device.
     * @return The data byte read from the specified address.
     */
    uint8_t readByte(uint8_t address);

    /**
     * Performs an SPI transfer, sending and receiving data.
     *
     * This function transfers data to and from the SPI device. It sends the specified `tx` (transmit)
     * buffer and receives data into the `rx` (receive) buffer. The function uses the `ioctl` system
     * call with the `SPI_IOC_MESSAGE` command to initiate the transfer. If the device is not open or
     * the transfer fails, an exception is thrown.
     *
     * @param tx A pointer to the transmit buffer containing the data to send to the SPI device.
     * @param rx A pointer to the receive buffer where data read from the SPI device will be stored.
     * @param length The length of the data to transfer (in bytes).
     *
     * @throws std::runtime_error If the SPI device is not open or if the transfer fails, indicating
     *         an error during the SPI communication.
     */
    void spiTransfer(const uint8_t *tx, uint8_t *rx, size_t length);

    /**
     * Closes the SPI device and releases the associated resources.
     *
     * This function closes the SPI device by calling the `close` system call on the SPI file descriptor
     * (`spi_fd`). It sets `spi_fd` to -1 to indicate that the device has been closed. If the device is
     * already closed (i.e., `spi_fd` is negative), the function does nothing.
     */
    void closeDevice();

private:
    int spi_fd;
    uint8_t mode;
    uint8_t bits;
    uint32_t speed;

    static constexpr uint8_t DefaultBitsPerWord = 8;
    static constexpr uint32_t DefaultSpeedHz = 1'000'000;
    static constexpr uint8_t DefaultMode = 0;
};

#endif // SPICONTROLLER_HPP
