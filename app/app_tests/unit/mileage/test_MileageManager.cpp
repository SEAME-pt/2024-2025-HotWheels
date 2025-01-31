#include <QSignalSpy>
#include "MileageManager.hpp"
#include "MockMileageCalculator.hpp"
#include "MockMileageFileHandler.hpp"
#include <gmock/gmock.h>
#include <gtest/gtest.h>

using ::testing::_;
using ::testing::NiceMock;
using ::testing::Return;

class MileageManagerTest : public ::testing::Test
{
protected:
    NiceMock<MockMileageCalculator> mockCalculator;
    NiceMock<MockMileageFileHandler> mockFileHandler;
    QString testFilePath = "test_mileage.txt";
    MileageManager *mileageManager;

    void SetUp() override
    {
        mileageManager = new MileageManager(testFilePath, &mockCalculator, &mockFileHandler);
    }

    void TearDown() override { delete mileageManager; }
};

TEST_F(MileageManagerTest, Initialize_LoadsMileageFromFile)
{
    EXPECT_CALL(mockFileHandler, readMileage()).WillOnce(Return(123.45)); // Simulate stored mileage

    // Allow writeMileage to be called more than once due to shutdown
    EXPECT_CALL(mockFileHandler, writeMileage(123.45)).WillRepeatedly(Return());

    mileageManager->initialize();
    mileageManager->saveMileage();
}

TEST_F(MileageManagerTest, OnSpeedUpdated_CallsCalculator)
{
    EXPECT_CALL(mockCalculator, addSpeed(50.0)).Times(1);
    mileageManager->onSpeedUpdated(50.0);
}

TEST_F(MileageManagerTest, UpdateMileage_EmitsMileageUpdatedSignal)
{
    EXPECT_CALL(mockCalculator, calculateDistance()).WillOnce(Return(10.5));

    QSignalSpy spy(mileageManager, &MileageManager::mileageUpdated);
    ASSERT_TRUE(spy.isValid());

    mileageManager->updateMileage();

    ASSERT_EQ(spy.count(), 1);
    QList<QVariant> arguments = spy.takeFirst();
    EXPECT_DOUBLE_EQ(arguments.at(0).toDouble(), 10.5);
}

TEST_F(MileageManagerTest, SaveMileage_CallsFileHandler)
{
    EXPECT_CALL(mockFileHandler, writeMileage(200.0)).WillRepeatedly(Return());

    EXPECT_CALL(mockCalculator, calculateDistance()).WillOnce(Return(50.0));
    mileageManager->updateMileage(); // Adds 50.0

    EXPECT_CALL(mockCalculator, calculateDistance()).WillOnce(Return(150.0));
    mileageManager->updateMileage(); // Adds another 150.0

    mileageManager->saveMileage();
}
