#include <QCoreApplication>
#include <QSignalSpy>
#include "MileageCalculator.hpp"
#include "MileageFileHandler.hpp"
#include "MileageManager.hpp"
#include <gtest/gtest.h>

class MileageManagerTest : public ::testing::Test
{
protected:
    static QCoreApplication *app;
    MileageManager *mileageManager;
    IMileageCalculator *calculator;
    IMileageFileHandler *fileHandler;

    static void SetUpTestSuite()
    {
        int argc = 0;
        char *argv[] = {nullptr};
        app = new QCoreApplication(argc, argv);
    }

    static void TearDownTestSuite() { delete app; }

    void SetUp() override
    {
        calculator = new MileageCalculator();
        fileHandler = new MileageFileHandler("/home/hotweels/app_data/test_mileage.json");
        mileageManager = new MileageManager("/home/hotweels/app_data/test_mileage.json",
                                            calculator,
                                            fileHandler);
    }

    void TearDown() override
    {
        delete mileageManager;
        delete calculator;
        delete fileHandler;
    }
};

QCoreApplication *MileageManagerTest::app = nullptr;

TEST_F(MileageManagerTest, ForwardMileageData)
{
    QSignalSpy spy(mileageManager, &MileageManager::mileageUpdated);

    mileageManager->onSpeedUpdated(10.0f);
    QCoreApplication::processEvents();

    ASSERT_GT(spy.count(), 0);
    QList<QVariant> args = spy.takeFirst();
    EXPECT_DOUBLE_EQ(args.at(0).toDouble(), 10.0);
}

TEST_F(MileageManagerTest, InitializeMileageManager)
{
    mileageManager->initialize();
    QSignalSpy spy(mileageManager, &MileageManager::mileageUpdated);

    QCoreApplication::processEvents();
    ASSERT_GT(spy.count(), 0);
    QList<QVariant> args = spy.takeFirst();
    EXPECT_DOUBLE_EQ(args.at(0).toDouble(), 0.0);
}

TEST_F(MileageManagerTest, UpdateMileageOnSpeedUpdate)
{
    QSignalSpy spy(mileageManager, &MileageManager::mileageUpdated);

    mileageManager->onSpeedUpdated(10.0f);
    QCoreApplication::processEvents();

    ASSERT_GT(spy.count(), 0);
    QList<QVariant> args = spy.takeFirst();
    EXPECT_DOUBLE_EQ(args.at(0).toDouble(), 10.0);
}

TEST_F(MileageManagerTest, SaveMileage)
{
    mileageManager->onSpeedUpdated(5.0f);
    QCoreApplication::processEvents();

    mileageManager->saveMileage();
    QCoreApplication::processEvents();

    double savedMileage = mileageManager->getTotalMileage();
    EXPECT_DOUBLE_EQ(savedMileage, 5.0);
}

TEST_F(MileageManagerTest, UpdateTimerInterval)
{
    QSignalSpy spy(mileageManager, &MileageManager::mileageUpdated);

    mileageManager->initialize();
    QTimer::singleShot(1000, QCoreApplication::instance(), &QCoreApplication::quit);
    QCoreApplication::processEvents();

    ASSERT_GT(spy.count(), 0);
}

TEST_F(MileageManagerTest, ShutdownMileageManager)
{
    mileageManager->initialize();
    mileageManager->shutdown();

    double finalMileage = mileageManager->getTotalMileage();
    EXPECT_DOUBLE_EQ(finalMileage, 0.0);
}
