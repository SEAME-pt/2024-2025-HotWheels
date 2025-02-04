#include <QCoreApplication>
#include <QSignalSpy>
#include "BatteryController.hpp"
#include "SystemCommandExecutor.hpp"
#include "SystemInfoProvider.hpp"
#include "SystemManager.hpp"
#include <gtest/gtest.h>

class SystemManagerTest : public ::testing::Test
{
protected:
    static QCoreApplication *app;
    SystemManager *systemManager;
    IBatteryController *batteryController;
    ISystemInfoProvider *systemInfoProvider;
    ISystemCommandExecutor *systemCommandExecutor;

    static void SetUpTestSuite()
    {
        int argc = 0;
        char *argv[] = {nullptr};
        app = new QCoreApplication(argc, argv);
    }

    static void TearDownTestSuite() { delete app; }

    void SetUp() override
    {
        batteryController = new BatteryController();
        systemInfoProvider = new SystemInfoProvider();
        systemCommandExecutor = new SystemCommandExecutor();
        systemManager = new SystemManager(batteryController,
                                          systemInfoProvider,
                                          systemCommandExecutor);
    }

    void TearDown() override
    {
        delete systemManager;
        delete batteryController;
        delete systemInfoProvider;
        delete systemCommandExecutor;
    }
};

QCoreApplication *SystemManagerTest::app = nullptr;

TEST_F(SystemManagerTest, UpdateTimeSignal)
{
    QSignalSpy spy(systemManager, &SystemManager::timeUpdated);

    systemManager->initialize();
    QCoreApplication::processEvents();

    ASSERT_GT(spy.count(), 0);
    QList<QVariant> args = spy.takeFirst();
    EXPECT_FALSE(args.isEmpty());
}

TEST_F(SystemManagerTest, UpdateWifiStatusSignal)
{
    QSignalSpy spy(systemManager, &SystemManager::wifiStatusUpdated);

    systemManager->initialize();
    QCoreApplication::processEvents();

    ASSERT_GT(spy.count(), 0);
    QList<QVariant> args = spy.takeFirst();
    EXPECT_FALSE(args.isEmpty());
}

TEST_F(SystemManagerTest, UpdateTemperatureSignal)
{
    QSignalSpy spy(systemManager, &SystemManager::temperatureUpdated);

    systemManager->initialize();
    QCoreApplication::processEvents();

    ASSERT_GT(spy.count(), 0);
    QList<QVariant> args = spy.takeFirst();
    EXPECT_FALSE(args.isEmpty());
}

TEST_F(SystemManagerTest, UpdateBatteryPercentageSignal)
{
    QSignalSpy spy(systemManager, &SystemManager::batteryPercentageUpdated);

    systemManager->initialize();
    QCoreApplication::processEvents();

    ASSERT_GT(spy.count(), 0);
    QList<QVariant> args = spy.takeFirst();
    EXPECT_GE(args.at(0).toFloat(), 0.0f);
    EXPECT_LE(args.at(0).toFloat(), 100.0f);
}

TEST_F(SystemManagerTest, UpdateIpAddressSignal)
{
    QSignalSpy spy(systemManager, &SystemManager::ipAddressUpdated);

    systemManager->initialize();
    QCoreApplication::processEvents();

    ASSERT_GT(spy.count(), 0);
    QList<QVariant> args = spy.takeFirst();
    EXPECT_FALSE(args.isEmpty());
}

TEST_F(SystemManagerTest, ShutdownSystemManager)
{
    systemManager->initialize();
    systemManager->shutdown();

    EXPECT_EQ(systemManager->getTimeTimer().isActive(), false);
    EXPECT_EQ(systemManager->getStatusTimer().isActive(), false);
}
