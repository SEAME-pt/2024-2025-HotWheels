/**
 * @file test_CanBusManager.cpp
 * @brief Unit tests for the CanBusManager class.
 * @version 0.1
 * @date 2025-01-30
 * @author Félix LE BIHAN (@Fle-bihh)
 * @author Ricardo Melo (@reomelo)
 * @author Tiago Pereira (@t-pereira06)
 * @author Michel Batista (@MicchelFAB)
 *
 * @details This file contains unit tests for the CanBusManager class, using
 * Google Test and Google Mock frameworks.
 *
 * @copyright Copyright (c) 2025
 */

#include "CanBusManager.hpp"
#include "MockMCP2515Controller.hpp"
#include <QSignalSpy>
#include <gmock/gmock.h>
#include <gtest/gtest.h>

using ::testing::_;
using ::testing::Return;

/**
 * @class CanBusManagerTest
 * @brief Test fixture for testing the CanBusManager class.
 *
 * @details This class sets up the necessary mock objects and provides setup and
 * teardown methods for each test.
 */
class CanBusManagerTest : public ::testing::Test {
protected:
  /**
   * @brief Set up the test environment.
   *
   * @details This method is called before each test to set up the necessary
   * objects.
   */
  void SetUp() override {
    mockMcpController = new MockMCP2515Controller();
    manager = new CanBusManager(mockMcpController);
  }

  /**
   * @brief Tear down the test environment.
   *
   * @details This method is called after each test to clean up the objects
   * created in SetUp().
   */
  void TearDown() override {
    if (manager != nullptr) {
      delete manager;
    }
    if (mockMcpController != nullptr) {
      delete mockMcpController;
    }
  }


  /** @brief Mocked MCP2515 controller. */
  IMCP2515Controller *mockMcpController;
  /** @brief CanBusManager object. */
  CanBusManager *manager;
};

/**
 * @test Tests if the `speedUpdated` signal is emitted correctly.
 * @brief Ensures that the speed signal is emitted with the correct value.
 *
 * @details This test uses `QSignalSpy` to verify that `speedUpdated` emits the
 * expected speed value when the mock controller triggers the signal.
 *
 * @see CanBusManager::speedUpdated
 */
TEST_F(CanBusManagerTest, SpeedSignalEmitsCorrectly) {
  QSignalSpy speedSpy(manager, &CanBusManager::speedUpdated);

  float expectedSpeed = 150.0f;
  emit mockMcpController->speedUpdated(expectedSpeed);

  ASSERT_EQ(speedSpy.count(), 1);
  ASSERT_EQ(speedSpy.takeFirst().at(0).toFloat(), expectedSpeed);
}

/**
 * @test Tests if the `rpmUpdated` signal is emitted correctly.
 * @brief Ensures that the RPM signal emits the correct value.
 *
 * @details Similar to `SpeedSignalEmitsCorrectly`, this test verifies that
 *          `rpmUpdated` emits the expected RPM value.
 *
 * @see CanBusManager::rpmUpdated
 */
TEST_F(CanBusManagerTest, RpmSignalEmitsCorrectly) {
  QSignalSpy rpmSpy(manager, &CanBusManager::rpmUpdated);

  int expectedRpm = 2500;
  emit mockMcpController->rpmUpdated(expectedRpm);

  ASSERT_EQ(rpmSpy.count(), 1);
  ASSERT_EQ(rpmSpy.takeFirst().at(0).toInt(), expectedRpm);
}

/**
 * @test Tests `CanBusManager::initialize()` when `init()` fails.
 * @brief Ensures that `initialize()` fails when the controller fails to
 * initialize.
 *
 * @details This test sets up a mock expectation that `init()` will return
 * `false`, verifying that `initialize()` correctly fails in this case.
 *
 * @see CanBusManager::initialize
 */
TEST_F(CanBusManagerTest, InitializeFailsWhenControllerFails) {
  EXPECT_CALL(*static_cast<MockMCP2515Controller *>(mockMcpController), init())
      .WillOnce(Return(false));

  ASSERT_FALSE(manager->initialize());
}

/**
 * @test Tests `CanBusManager::initialize()` when `init()` succeeds.
 * @brief Ensures that `initialize()` succeeds when the controller initializes
 * successfully.
 *
 * @details This test sets up a mock expectation that `init()` will return
 * `true`, verifying that `initialize()` correctly returns `true`.
 *
 * @see CanBusManager::initialize
 */
TEST_F(CanBusManagerTest, InitializeSucceedsWhenControllerSucceeds) {
  EXPECT_CALL(*static_cast<MockMCP2515Controller *>(mockMcpController), init())
      .WillOnce(Return(true));

  ASSERT_TRUE(manager->initialize());
}

/**
 * @test Ensures that `stopReading()` is called when `CanBusManager` is
 * destroyed.
 * @brief Verifies that `stopReading()` is triggered in the destructor.
 *
 * @details This test initializes `CanBusManager`, then deletes it and confirms
 *          that `stopReading()` was called exactly once.
 *
 * @note This test indirectly verifies proper cleanup behavior in the
 * destructor.
 *
 * @see CanBusManager::~CanBusManager
 */
TEST_F(CanBusManagerTest, DestructorCallsStopReading) {
  EXPECT_CALL(*static_cast<MockMCP2515Controller *>(mockMcpController), init())
      .WillOnce(Return(true));
  EXPECT_CALL(*static_cast<MockMCP2515Controller *>(mockMcpController),
              stopReading())
      .Times(1);

  ASSERT_TRUE(manager->initialize());

  delete manager;
  manager = nullptr;

  SUCCEED();
}