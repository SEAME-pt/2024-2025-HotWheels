/*!
 * @file test_MileageFileHandler.cpp
 * @brief Unit tests for the MileageFileHandler class.
 * @version 0.1
 * @date 2025-02-12
 * @details This file contains unit tests for the MileageFileHandler class, using
 * Google Test and Google Mock frameworks.
 * @author Félix LE BIHAN (@Fle-bihh)
 * @author Tiago Pereira (@t-pereira06)
 * @author Ricardo Melo (@reomelo)
 * @author Michel Batista (@MicchelFAB)
 *
 * @copyright Copyright (c) 2025
 */

#include "MileageFileHandler.hpp"
#include "MockFileController.hpp"
#include <gmock/gmock.h>
#include <gtest/gtest.h>

using ::testing::_;
using ::testing::Return;

/*!
 * @class MileageFileHandlerTest
 * @brief Test fixture for testing the MileageFileHandler class.
 *
 * @details This class sets up the necessary objects and provides setup and
 * teardown methods for each test.
 */
class MileageFileHandlerTest : public ::testing::Test
{
protected:
	::testing::NiceMock<MockFileController> mockFileController;
	QString testFilePath = "test_mileage.txt";

	MileageFileHandler *mileageFileHandler;

	void SetUp() override {
		ON_CALL(mockFileController, exists(::testing::_)).WillByDefault(::testing::Return(false));
		ON_CALL(mockFileController, open(::testing::_, ::testing::_)).WillByDefault(::testing::Return(false));

		initFileHandler();
	}

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

/*!
 * @test Tests if the mileage file handler creates the file if it does not exist.
 * @brief Ensures that the mileage file handler creates the file if it does not exist.
 * @details This test verifies that the mileage file handler creates the file if it does not exist.
 *
 * @see MileageFileHandler::ensureFileExists
*/
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

/*!
 * @test Tests if the mileage file handler does not create the file if it already exists.
 * @brief Ensures that the mileage file handler does not create the file if it already exists.
 * @details This test verifies that the mileage file handler does not create the file if it already exists.
 *
 * @see MileageFileHandler::ensureFileExists
*/
TEST_F(MileageFileHandlerTest, ReadMileage_ValidNumber_ReturnsParsedValue)
{
	EXPECT_CALL(mockFileController, open(testing::_, QIODevice::ReadOnly | QIODevice::Text))
		.WillOnce(Return(true)); // Simulate file opening

	EXPECT_CALL(mockFileController, read(testing::_))
		.WillOnce(Return("123.45")); // Simulate reading mileage

	double mileage = mileageFileHandler->readMileage();
	EXPECT_DOUBLE_EQ(mileage, 123.45);
}

/*!
 * @test Tests if the mileage file handler reads zero mileage if the file is empty.
 * @brief Ensures that the mileage file handler reads zero mileage if the file is empty.
 * @details This test verifies that the mileage file handler reads zero mileage if the file is empty.
 *
 * @see MileageFileHandler::readMileage
*/
TEST_F(MileageFileHandlerTest, ReadMileage_InvalidNumber_ReturnsZero)
{
	EXPECT_CALL(mockFileController, open(testing::_, QIODevice::ReadOnly | QIODevice::Text))
		.WillOnce(Return(true)); // Simulate file opening

	EXPECT_CALL(mockFileController, read(testing::_))
		.WillOnce(Return("INVALID")); // Simulate reading invalid mileage

	double mileage = mileageFileHandler->readMileage();
	EXPECT_DOUBLE_EQ(mileage, 0.0);
}

/*!
 * @test Tests if the mileage file handler writes the mileage to the file.
 * @brief Ensures that the mileage file handler writes the mileage to the file.
 * @details This test verifies that the mileage file handler writes the mileage to the file.
 *
 * @see MileageFileHandler::writeMileage
*/
TEST_F(MileageFileHandlerTest, WriteMileage_ValidNumber_WritesToFile)
{
	EXPECT_CALL(mockFileController, open(testing::_, QIODevice::WriteOnly | QIODevice::Text))
		.WillOnce(Return(true)); // Simulate file opening

	EXPECT_CALL(mockFileController, write(testing::_, QString("789.12")))
		.WillOnce(Return(true)); // Simulate successful write

	mileageFileHandler->writeMileage(789.12);
}
