#include <QCoreApplication>
#include <QDebug>
#include <QSignalSpy>
#include "CanBusManager.hpp"
#include "MCP2515Controller.hpp"
#include <gtest/gtest.h>

class CanBusManagerTest : public ::testing::Test
{
protected:
    static QCoreApplication *app;
    CanBusManager *canBusManager;
    IMCP2515Controller *controller;

    static void SetUpTestSuite()
    {
        int argc = 0;
        char *argv[] = {nullptr};
        app = new QCoreApplication(argc, argv);
    }

    static void TearDownTestSuite() { delete app; }

    void SetUp() override
    {
        controller = new MCP2515Controller("/dev/spidev0.0");
        ASSERT_NE(controller, nullptr);

        canBusManager = new CanBusManager(controller);
        ASSERT_NE(canBusManager, nullptr);
    }

    void TearDown() override
    {
        delete canBusManager;
        delete controller;
    }
};

// Initialize static member
QCoreApplication *CanBusManagerTest::app = nullptr;

// 🚗 **Test: Forward Speed Data**
TEST_F(CanBusManagerTest, ForwardSpeedDataFromMCP2515)
{
    QSignalSpy spy(canBusManager, &CanBusManager::speedUpdated);

    emit controller->speedUpdated(88.8f);
    QCoreApplication::processEvents();

    ASSERT_GT(spy.count(), 0) << "Signal was not emitted!";
    QList<QVariant> args = spy.takeFirst();
    EXPECT_FLOAT_EQ(args.at(0).toFloat(), 88.8f);
}

// 🔄 **Test: Forward RPM Data**
TEST_F(CanBusManagerTest, ForwardRpmDataFromMCP2515)
{
    QSignalSpy spy(canBusManager, &CanBusManager::rpmUpdated);

    emit controller->rpmUpdated(4500);
    QCoreApplication::processEvents();

    ASSERT_GT(spy.count(), 0) << "Signal was not emitted!";
    QList<QVariant> args = spy.takeFirst();
    EXPECT_EQ(args.at(0).toInt(), 4500);
}

// 🚀 **Test: Initialization**
TEST_F(CanBusManagerTest, InitializeCanBusManager)
{
    ASSERT_TRUE(canBusManager->initialize()) << "Initialization failed!";
    ASSERT_NE(canBusManager->getThread(), nullptr) << "Thread not created!";
    ASSERT_TRUE(canBusManager->getThread()->isRunning()) << "Thread did not start!";
}

TEST_F(CanBusManagerTest, ManagerCleanUpBehavior)
{
    CanBusManager *tmpManager = new CanBusManager("/dev/spidev0.0");
    ASSERT_NE(tmpManager, nullptr);

    ASSERT_EQ(tmpManager->getThread(), nullptr) << "Thread created too soon!";

    tmpManager->initialize();

    ASSERT_NE(tmpManager->getThread(), nullptr) << "Thread not created!";
    ASSERT_TRUE(tmpManager->getThread()->isRunning()) << "Thread did not start!";

    delete tmpManager;

    ASSERT_EQ(tmpManager->getThread(), nullptr) << "Thread not deleted!";
}
