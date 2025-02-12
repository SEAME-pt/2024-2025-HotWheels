/*!
 * @file IMileageCalculator.hpp
 * @brief Definition of the IMileageCalculator interface.
 * @version 0.1
 * @date 2025-02-12
 * @details This file contains the definition of the IMileageCalculator interface, which
 * is responsible for calculating the mileage of a vehicle.
 * @author Félix LE BIHAN (@Fle-bihh)
 * @author Tiago Pereira (@t-pereira06)
 * @author Ricardo Melo (@reomelo)
 * @author Michel Batista (@MicchelFAB)
 *
 * @copyright Copyright (c) 2025
 */

#ifndef IMILEAGECALCULATOR_HPP
#define IMILEAGECALCULATOR_HPP

/*!
 * @brief Interface for calculating the mileage of a vehicle.
 * @class IMileageCalculator
 */
class IMileageCalculator
{
public:
    virtual ~IMileageCalculator() = default;
    virtual void addSpeed(float speed) = 0;
    virtual double calculateDistance() = 0;
};

#endif // IMILEAGECALCULATOR_HPP
