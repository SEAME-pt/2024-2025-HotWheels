#include <QElapsedTimer>
#include <QTest>
#include "MileageCalculator.hpp"
#include <gtest/gtest.h>

class MileageCalculatorTest : public ::testing::Test
{
protected:
    MileageCalculator calculator;
};

TEST_F(MileageCalculatorTest, AddSpeed_DoesNotCrash)
{
    calculator.addSpeed(50.0);
    calculator.addSpeed(80.0);
    EXPECT_NO_THROW(calculator.calculateDistance());
}

TEST_F(MileageCalculatorTest, CalculateDistance_NoSpeeds_ReturnsZero)
{
    EXPECT_DOUBLE_EQ(calculator.calculateDistance(), 0.0);
}

TEST_F(MileageCalculatorTest, CalculateDistance_ZeroSpeed_ReturnsZero)
{
    calculator.addSpeed(0.0);
    QTest::qWait(100); // Wait before next speed is logged
    calculator.addSpeed(0.0);
    EXPECT_DOUBLE_EQ(calculator.calculateDistance(), 0.0);
}

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
