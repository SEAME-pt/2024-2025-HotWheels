/*!
 * @file test_MileageCalculator.cpp
 * @brief Unit tests for the MileageCalculator class.
 * @version 0.1
 * @date 2025-02-12
 * @details This file contains unit tests for the MileageCalculator class, using
 * Google Test and Google Mock frameworks.
 * @author Félix LE BIHAN (@Fle-bihh)
 * @author Tiago Pereira (@t-pereira06)
 * @author Ricardo Melo (@reomelo)
 * @author Michel Batista (@MicchelFAB)
 *
 * @copyright Copyright (c) 2025
 */

#include <QElapsedTimer>
#include <QTest>
#include "MileageCalculator.hpp"
#include <gtest/gtest.h>

/*!
 * @class MileageCalculatorTest
 * @brief Test fixture for testing the MileageCalculator class.
 *
 * @details This class sets up the necessary objects and provides setup and
 * teardown methods for each test.
 */
class MileageCalculatorTest : public ::testing::Test
{
protected:
	MileageCalculator calculator;
};

/*!
 * @test Tests if the mileage calculator does not crash when adding speeds.
 * @brief Ensures that the mileage calculator does not crash when adding speeds.
 *
 * @details This test verifies that the mileage calculator does not crash when
 * adding speeds.
 *
 * @see MileageCalculator::addSpeed
 */
TEST_F(MileageCalculatorTest, AddSpeed_DoesNotCrash)
{
	calculator.addSpeed(50.0);
	calculator.addSpeed(80.0);
	EXPECT_NO_THROW(calculator.calculateDistance());
}

/*!
 * @test Tests if the mileage calculator calculates the distance correctly.
 * @brief Ensures that the mileage calculator calculates the distance correctly.
 *
 * @details This test verifies that the mileage calculator calculates the distance
 * correctly.
 *
 * @see MileageCalculator::calculateDistance
 */
TEST_F(MileageCalculatorTest, CalculateDistance_NoSpeeds_ReturnsZero)
{
	EXPECT_DOUBLE_EQ(calculator.calculateDistance(), 0.0);
}

/*!
 * @test Tests if the mileage calculator calculates the distance correctly.
 * @brief Ensures that the mileage calculator calculates the distance correctly.
 *
 * @details This test verifies that the mileage calculator calculates the distance
 * correctly.
 *
 * @see MileageCalculator::calculateDistance
 */
TEST_F(MileageCalculatorTest, CalculateDistance_ZeroSpeed_ReturnsZero)
{
	calculator.addSpeed(0.0);
	QTest::qWait(100); // Wait before next speed is logged
	calculator.addSpeed(0.0);
	EXPECT_DOUBLE_EQ(calculator.calculateDistance(), 0.0);
}

/*!
 * @test Tests if the mileage calculator calculates the distance correctly.
 * @brief Ensures that the mileage calculator calculates the distance correctly.
 *
 * @details This test verifies that the mileage calculator calculates the distance
 * correctly.
 *
 * @see MileageCalculator::calculateDistance
 */
TEST_F(MileageCalculatorTest, CalculateDistance_BasicCalculation)
{
	QElapsedTimer timer;

	timer.start();
	QTest::qWait(100);                 // Reduced wait time
	calculator.addSpeed(60.0);         // 60 km/h (16.67 m/s)
	qint64 elapsed1 = timer.restart(); // Measure interval before next speed

	QTest::qWait(100);                 // Reduced wait time
	calculator.addSpeed(90.0);         // 90 km/h (25.0 m/s)
	qint64 elapsed2 = timer.restart(); // Measure interval before next speed

	double distance = calculator.calculateDistance();

	// Compute expected distance using elapsed time BEFORE each speed entry
	double expected = ((60.0 / 3.6) * (elapsed1 / 1000.0)) + ((90.0 / 3.6) * (elapsed2 / 1000.0));

	EXPECT_NEAR(distance, expected, 0.1);
}

/*!
 * @test Tests if the mileage calculator calculates the distance correctly.
 * @brief Ensures that the mileage calculator calculates the distance correctly.
 *
 * @details This test verifies that the mileage calculator calculates the distance
 * correctly.
 *
 * @see MileageCalculator::calculateDistance
 */
TEST_F(MileageCalculatorTest, CalculateDistance_MultipleSpeeds)
{
	QElapsedTimer timer;

	timer.start();
	QTest::qWait(50);
	calculator.addSpeed(30.0); // 30 km/h (8.33 m/s)
	qint64 elapsed1 = timer.restart();

	QTest::qWait(75);
	calculator.addSpeed(50.0); // 50 km/h (13.89 m/s)
	qint64 elapsed2 = timer.restart();

	QTest::qWait(50);
	calculator.addSpeed(80.0); // 80 km/h (22.22 m/s)
	qint64 elapsed3 = timer.restart();

	QTest::qWait(75);
	calculator.addSpeed(100.0); // 100 km/h (27.78 m/s)
	qint64 elapsed4 = timer.restart();

	double distance = calculator.calculateDistance();

	// Compute expected distance using measured intervals
	double expected = ((30.0 / 3.6) * (elapsed1 / 1000.0)) + ((50.0 / 3.6) * (elapsed2 / 1000.0))
					  + ((80.0 / 3.6) * (elapsed3 / 1000.0))
					  + ((100.0 / 3.6) * (elapsed4 / 1000.0));

	EXPECT_NEAR(distance, expected, 0.1);
}

TEST_F(MileageCalculatorTest, InterfaceUsage_CoversIMileageCalculator)
{
	IMileageCalculator* interface = new MileageCalculator();

	interface->addSpeed(50.0);
	double distance = interface->calculateDistance();

	EXPECT_GT(distance, 0.0);
	delete interface;
}
