#include <QSignalSpy>
#include "CanBusManager.hpp"
#include "MockMCP2515Controller.hpp"
#include <gmock/gmock.h>
#include <gtest/gtest.h>

using ::testing::_;
using ::testing::Return;

class CanBusManagerTest : public ::testing::Test
{
protected:
    IMCP2515Controller *mockMcpController;
    CanBusManager *manager;

    void SetUp() override
    {
        mockMcpController = new MockMCP2515Controller();
        manager = new CanBusManager(mockMcpController);
    }

    void TearDown() override
    {
        if (manager) {
            delete manager;
        }
        if (mockMcpController) {
            delete mockMcpController;
        }
    }
};

TEST_F(CanBusManagerTest, SpeedSignalEmitsCorrectly)
{
    QSignalSpy speedSpy(manager, &CanBusManager::speedUpdated);

    float expectedSpeed = 150.0f;
    emit mockMcpController->speedUpdated(expectedSpeed);

    ASSERT_EQ(speedSpy.count(), 1);
    ASSERT_EQ(speedSpy.takeFirst().at(0).toFloat(), expectedSpeed);
}

TEST_F(CanBusManagerTest, RpmSignalEmitsCorrectly)
{
    QSignalSpy rpmSpy(manager, &CanBusManager::rpmUpdated);

    int expectedRpm = 2500;
    emit mockMcpController->rpmUpdated(expectedRpm);

    ASSERT_EQ(rpmSpy.count(), 1);
    ASSERT_EQ(rpmSpy.takeFirst().at(0).toInt(), expectedRpm);
}

TEST_F(CanBusManagerTest, InitializeFailsWhenControllerFails)
{
    EXPECT_CALL(*static_cast<MockMCP2515Controller *>(mockMcpController), init())
        .WillOnce(Return(false));

    ASSERT_FALSE(manager->initialize());
}

TEST_F(CanBusManagerTest, InitializeSucceedsWhenControllerSucceeds)
{
    EXPECT_CALL(*static_cast<MockMCP2515Controller *>(mockMcpController), init())
        .WillOnce(Return(true));

    ASSERT_TRUE(manager->initialize());
}

TEST_F(CanBusManagerTest, DestructorCallsStopReading)
{
    EXPECT_CALL(*static_cast<MockMCP2515Controller *>(mockMcpController), init())
        .WillOnce(Return(true));
    EXPECT_CALL(*static_cast<MockMCP2515Controller *>(mockMcpController), stopReading()).Times(1);
    ASSERT_TRUE(manager->initialize());

    delete manager;
    manager = nullptr;

    SUCCEED();
}
