#include "MileageFileHandler.hpp"
#include "MockFileController.hpp"
#include <gmock/gmock.h>
#include <gtest/gtest.h>

using ::testing::_;
using ::testing::Return;

class MileageFileHandlerTest : public ::testing::Test
{
protected:
    MockFileController mockFileController;
    QString testFilePath = "test_mileage.txt";

    MileageFileHandler *mileageFileHandler;

    void SetUp() override { initFileHandler(); }

    void TearDown() override { delete mileageFileHandler; }

    void initFileHandler()
    {
        mileageFileHandler = new MileageFileHandler(
            testFilePath,
            [this](QFile &file, QIODevice::OpenMode mode) {
                return mockFileController.open(file, mode);
            },
            [this](QFile &file) { return mockFileController.read(file); },
            [this](QFile &file, const QString &data) {
                return mockFileController.write(file, data);
            },
            [this](const QString &path) { return mockFileController.exists(path); });
    };
};

TEST_F(MileageFileHandlerTest, EnsureFileExists_FileDoesNotExist_CreatesFileWithZeroMileage)
{
    // Ensure file does NOT exist
    EXPECT_CALL(mockFileController, exists(testFilePath)).WillOnce(Return(false));

    // Expect file to open for writing
    EXPECT_CALL(mockFileController, open(_, QIODevice::WriteOnly | QIODevice::Text))
        .WillOnce(Return(true));

    // Expect writing "0.0" to file
    EXPECT_CALL(mockFileController, write(_, QString("0.0"))).WillOnce(Return(true));

    // Manually trigger ensureFileExists()
    MileageFileHandler *tmpHandler = new MileageFileHandler(
        testFilePath,
        [this](QFile &file, QIODevice::OpenMode mode) {
            return mockFileController.open(file, mode);
        },
        [this](QFile &file) { return mockFileController.read(file); },
        [this](QFile &file, const QString &data) { return mockFileController.write(file, data); },
        [this](const QString &path) { return mockFileController.exists(path); });

    delete tmpHandler;
}

TEST_F(MileageFileHandlerTest, ReadMileage_ValidNumber_ReturnsParsedValue)
{
    EXPECT_CALL(mockFileController, open(testing::_, QIODevice::ReadOnly | QIODevice::Text))
        .WillOnce(Return(true)); // Simulate file opening

    EXPECT_CALL(mockFileController, read(testing::_))
        .WillOnce(Return("123.45")); // Simulate reading mileage

    double mileage = mileageFileHandler->readMileage();
    EXPECT_DOUBLE_EQ(mileage, 123.45);
}

TEST_F(MileageFileHandlerTest, ReadMileage_InvalidNumber_ReturnsZero)
{
    EXPECT_CALL(mockFileController, open(testing::_, QIODevice::ReadOnly | QIODevice::Text))
        .WillOnce(Return(true)); // Simulate file opening

    EXPECT_CALL(mockFileController, read(testing::_))
        .WillOnce(Return("INVALID")); // Simulate reading invalid mileage

    double mileage = mileageFileHandler->readMileage();
    EXPECT_DOUBLE_EQ(mileage, 0.0);
}

TEST_F(MileageFileHandlerTest, WriteMileage_ValidNumber_WritesToFile)
{
    EXPECT_CALL(mockFileController, open(testing::_, QIODevice::WriteOnly | QIODevice::Text))
        .WillOnce(Return(true)); // Simulate file opening

    EXPECT_CALL(mockFileController, write(testing::_, QString("789.12")))
        .WillOnce(Return(true)); // Simulate successful write

    mileageFileHandler->writeMileage(789.12);
}
