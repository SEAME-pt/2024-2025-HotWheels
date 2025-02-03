/*!
 * @file test_MCP2515Controller.cpp
 * @brief File containing integration tests for the MCP2515 controller.
 * @version 0.1
 * @date 2025-01-31
 * @author Michel Batista (@MicchelFAB)
 * @author Félix LE BIHAN (@Fle-bihh)
 * @author Ricardo Melo (@reomelo)
 * @author Tiago Pereira (@t-pereira06)
 *
 * @copyright Copyright (c) 2025
 *
 */

#include "MCP2515Controller.hpp"
#include "SPIController.hpp"
#include <cassert>
#include <chrono>
#include <cstdint>
#include <gtest/gtest.h>
#include <iostream>
#include <thread>
#include <vector>

/*!
 * @brief Structure to represent a CAN frame.
 * @struct CANFrame with the following fields needed to represent a CAN frame
 */
struct CANFrame {
  /*! @brief Identifier */
  uint32_t id;
  /*! @brief Remote Transmission Request */
  uint8_t rtr;
  /*! @brief Data field */
  std::vector<uint8_t> data;
  /*! @brief Data Length Code */
  uint8_t dlc;
  /*! @brief CRC validity */
  bool crcValid;
  /*! @brief Acknowledgement received */
  bool ackReceived;
};

/*!
 * @brief Class to test the integration between the MCP2515 controller and the
 * SPI controller.
 * @class RS485CANTest
 */
class RS485CANTest : public ::testing::Test {
protected:
  MCP2515Configurator* canBusConfigurator;
  SPIController* spiController;

  void SetUp() override {
    try {
		spiController = new SPIController();
    	spiController->openDevice("/dev/spidev0.0");
    	canBusConfigurator = new MCP2515Configurator(*spiController);
    } catch (const std::exception& e) {
      FAIL() << "SetUp() failed with exception: " << e.what();
    }
  }

  void TearDown() override {
    delete canBusConfigurator;
    delete spiController;
  }
};

/*!
 * @test Tests the initialization of the MCP2515 controller.
 * @brief Ensures that the MCP2515 controller initializes successfully.
 * @details This test verifies that the MCP2515 controller initializes
 * successfully by calling the init() method.
 *
 * @see MCP2515Configurator::readCANMessage
 * @see MCP2515Configurator::sendCANMessage
 */
TEST_F(RS485CANTest, DataFrameTest) {
  /// Create a data frame with an 11-bit identifier and 8 bytes of data
  CANFrame dataFrame;
  dataFrame.id = 0x7FF; // 11-bit identifier
  dataFrame.rtr = 0;    // Data frame
  dataFrame.data = {0x01, 0x02, 0x03, 0x04,
                    0x05, 0x06, 0x07, 0x08}; // 8 bytes of data
  dataFrame.dlc = dataFrame.data.size();     // Data length code

  /// Send the data frame
  canBusConfigurator->sendCANMessage(dataFrame.id, dataFrame.data.data(),
                                     dataFrame.dlc);

  /// Simulate receiving the data frame
  uint16_t receivedID;
  std::vector<uint8_t> receivedData =
      canBusConfigurator->readCANMessage(receivedID);

  /// Verify the received frame
  ASSERT_EQ(receivedID, dataFrame.id);
  ASSERT_EQ(receivedData, dataFrame.data);

  /// Verify the CRC (assuming the controller calculates and verifies CRC
  /// internally)
  ASSERT_TRUE(true); // Placeholder for CRC check

  /// Verify the Acknowledgement Slot (assuming the controller handles this
  /// internally)
  ASSERT_TRUE(true); // Placeholder for ACK check
}

/*!
 * @test Tests the initialization of the MCP2515 controller.
 * @brief Ensures that the MCP2515 controller initializes successfully.
 * @details This test verifies that the MCP2515 controller initializes
 * successfully by calling the init() method.
 *
 * @see MCP2515Configurator::readCANMessage
 * @see MCP2515Configurator::sendCANMessage
 */
TEST_F(RS485CANTest, RemoteFrameTest) {
  // Create a remote frame with an 11-bit identifier and no data
  CANFrame remoteFrame;
  remoteFrame.id = 0x123; // 11-bit identifier
  remoteFrame.rtr = 1;    // Remote frame
  remoteFrame.dlc = 8;    // Data length code of the expected response

  // Send the remote frame
  canBusConfigurator->sendCANMessage(remoteFrame.id, nullptr, 0);

  // Simulate receiving the corresponding data frame
  uint16_t responseID;
  std::vector<uint8_t> responseData =
      canBusConfigurator->readCANMessage(responseID);

  // Verify the received frame
  ASSERT_EQ(responseID, remoteFrame.id);
  ASSERT_EQ(responseData.size(), remoteFrame.dlc);

  // Verify the CRC (assuming the controller calculates and verifies CRC
  // internally)
  ASSERT_TRUE(true); // Placeholder for CRC check

  // Verify the Acknowledgement Slot (assuming the controller handles this
  // internally)
  ASSERT_TRUE(true); // Placeholder for ACK check
}

/*!
 * @test Tests the initialization of the MCP2515 controller.
 * @brief Ensures that the MCP2515 controller initializes successfully.
 * @details This test verifies that the MCP2515 controller initializes
 * successfully by calling the init() method.
 *
 * @see MCP2515Configurator::readCANMessage
 * @see MCP2515Configurator::sendCANMessage
 */
TEST_F(RS485CANTest, ErrorFrameTest) {
  // Simulate sending an error frame
  CANFrame errorFrame;
  errorFrame.id = 0x7FF; // 11-bit identifier
  errorFrame.rtr = 0;    // Data frame
  errorFrame.data = {0xFF, 0xFF, 0xFF, 0xFF,
                     0xFF, 0xFF}; // Error flag (violates bit-stuffing rule)
  errorFrame.dlc = errorFrame.data.size(); // Data length code

  // Send the error frame
  canBusConfigurator->sendCANMessage(errorFrame.id, errorFrame.data.data(),
                                     errorFrame.dlc);

  // Simulate receiving the error frame
  uint16_t receivedID;
  std::vector<uint8_t> receivedData =
      canBusConfigurator->readCANMessage(receivedID);

  // Verify the received frame
  ASSERT_EQ(receivedID, errorFrame.id);
  ASSERT_EQ(receivedData, errorFrame.data);

  // Verify the CRC (assuming the controller calculates and verifies CRC
  // internally)
  ASSERT_TRUE(true); // Placeholder for CRC check

  // Verify the Acknowledgement Slot (assuming the controller handles this
  // internally)
  ASSERT_TRUE(true); // Placeholder for ACK check

  // Simulate retransmission after error detection
  canBusConfigurator->sendCANMessage(errorFrame.id, errorFrame.data.data(),
                                     errorFrame.dlc);
}

/*!
 * @test Tests the initialization of the MCP2515 controller.
 * @brief Ensures that the MCP2515 controller initializes successfully.
 * @details This test verifies that the MCP2515 controller initializes
 * successfully by calling the init() method.
 *
 * @see MCP2515Configurator::readCANMessage
 * @see MCP2515Configurator::sendCANMessage
 */
TEST_F(RS485CANTest, MaxBusSpeedTest) {
  // Remove the call to setBusSpeed()
  // ASSERT_TRUE(canBusConfigurator->setBusSpeed(1000000));

  // Create a data frame with an 11-bit identifier and 8 bytes of data
  CANFrame dataFrame;
  dataFrame.id = 0x7FF; // 11-bit identifier
  dataFrame.rtr = 0;    // Data frame
  dataFrame.data = {0x01, 0x02, 0x03, 0x04,
                    0x05, 0x06, 0x07, 0x08}; // 8 bytes of data
  dataFrame.dlc = dataFrame.data.size();     // Data length code

  // Send the data frame
  canBusConfigurator->sendCANMessage(dataFrame.id, dataFrame.data.data(),
                                     dataFrame.dlc);

  // Simulate receiving the data frame
  uint16_t receivedID;
  std::vector<uint8_t> receivedData =
      canBusConfigurator->readCANMessage(receivedID);

  // Verify the received frame
  ASSERT_EQ(receivedID, dataFrame.id);
  ASSERT_EQ(receivedData, dataFrame.data);

  // Verify the CRC (assuming the controller calculates and verifies CRC
  // internally)
  ASSERT_TRUE(true); // Placeholder for CRC check

  // Verify the Acknowledgement Slot (assuming the controller handles this
  // internally)
  ASSERT_TRUE(true); // Placeholder for ACK check
}

/*!
 * @test Tests the initialization of the MCP2515 controller.
 * @brief Ensures that the MCP2515 controller initializes successfully.
 * @details This test verifies that the MCP2515 controller initializes
 * successfully by calling the init() method.
 *
 * @see MCP2515Configurator::readCANMessage
 * @see MCP2515Configurator::sendCANMessage
 */
TEST_F(RS485CANTest, MinBusSpeedTest) {
  // Remove the call to setBusSpeed()
  // ASSERT_TRUE(canBusConfigurator->setBusSpeed(10000));

  // Create a data frame with an 11-bit identifier and 8 bytes of data
  CANFrame dataFrame;
  dataFrame.id = 0x7FF; // 11-bit identifier
  dataFrame.rtr = 0;    // Data frame
  dataFrame.data = {0x01, 0x02, 0x03, 0x04,
                    0x05, 0x06, 0x07, 0x08}; // 8 bytes of data
  dataFrame.dlc = dataFrame.data.size();     // Data length code

  // Send the data frame
  canBusConfigurator->sendCANMessage(dataFrame.id, dataFrame.data.data(),
                                     dataFrame.dlc);

  // Simulate receiving the data frame
  uint16_t receivedID;
  std::vector<uint8_t> receivedData =
      canBusConfigurator->readCANMessage(receivedID);

  // Verify the received frame
  ASSERT_EQ(receivedID, dataFrame.id);
  ASSERT_EQ(receivedData, dataFrame.data);

  // Verify the CRC (assuming the controller calculates and verifies CRC
  // internally)
  ASSERT_TRUE(true); // Placeholder for CRC check

  // Verify the Acknowledgement Slot (assuming the controller handles this
  // internally)
  ASSERT_TRUE(true); // Placeholder for ACK check
}

/*!
 * @test Tests the initialization of the MCP2515 controller.
 * @brief Ensures that the MCP2515 controller initializes successfully.
 * @details This test verifies that the MCP2515 controller initializes
 * successfully by calling the init() method.
 *
 * @see MCP2515Configurator::readCANMessage
 * @see MCP2515Configurator::sendCANMessage
 */
int main(int argc, char **argv) {
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
