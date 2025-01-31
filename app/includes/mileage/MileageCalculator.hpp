#ifndef MILEAGECALCULATOR_HPP
#define MILEAGECALCULATOR_HPP

#include <QElapsedTimer>
#include <QList>

class MileageCalculator {
public:
  /**
   * Constructor for the MileageCalculator class.
   * Initializes the interval timer to track the time between speed
   * measurements.
   */
  MileageCalculator();

  /**
   * Destructor for the MileageCalculator class.
   * The destructor is implicitly defaulted as no resources need explicit
   * cleanup.
   */
  ~MileageCalculator() = default;

  /**
   * Adds a new speed measurement along with the time interval.
   * The speed and interval are stored for later calculation of total distance.
   *
   * @param speed The speed measurement in kilometers per hour.
   */
  void addSpeed(float speed);

  /**
   * Calculates the total distance traveled based on the stored speed values and
   * their corresponding intervals.
   *
   * @return The total distance traveled in meters.
   */
  double calculateDistance();

private:
  QList<QPair<float, qint64>> m_speedValues; /**< List of speed and interval
                                                pairs to store measurements. */
  QElapsedTimer m_intervalTimer; /**< Timer used to track the time intervals
                                    between speed measurements. */
};

#endif // MILEAGECALCULATOR_HPP
