/**
 * @file MileageCalculator.hpp
 * @brief Definition of the MileageCalculator class.
 * @version 0.1
 * @date 2025-01-31
 * @details This file contains the definition of the MileageCalculator class,
 * which is responsible for calculating the total distance traveled based on
 * speed measurements.
 * @author Félix LE BIHAN (@Fle-bihh)
 * @author Tiago Pereira (@t-pereira06)
 * @author Ricardo Melo (@reomelo)
 * @author Michel Batista (@MicchelFAB)
 *
 * @copyright Copyright (c) 2025
 */

#ifndef MILEAGECALCULATOR_HPP
#define MILEAGECALCULATOR_HPP

#include <QElapsedTimer>
#include <QList>

/**
 * @brief Class that calculates the total distance traveled based on speed
 * measurements.
 * @class MileageCalculator
 */
class MileageCalculator {
public:
  MileageCalculator();
  ~MileageCalculator() = default;
  void addSpeed(float speed);
  double calculateDistance();

private:
  /** @brief List of speed and interval pairs to store measurements. */
  QList<QPair<float, qint64>> m_speedValues;
  /** @brief Timer used to track the time intervals between speed measurements.
   */
  QElapsedTimer m_intervalTimer;
};

#endif // MILEAGECALCULATOR_HPP
