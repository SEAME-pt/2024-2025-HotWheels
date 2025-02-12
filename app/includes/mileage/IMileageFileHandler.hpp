/*!
 * @file IMileageFileHandler.hpp
 * @brief Definition of the IMileageFileHandler interface.
 * @version 0.1
 * @date 2025-02-12
 * @details This file contains the definition of the IMileageFileHandler interface, which
 * is responsible for reading and writing the mileage of a vehicle to a file.
 * @author Félix LE BIHAN (@Fle-bihh)
 * @author Tiago Pereira (@t-pereira06)
 * @author Ricardo Melo (@reomelo)
 * @author Michel Batista (@MicchelFAB)
 *
 * @copyright Copyright (c) 2025
 */

#ifndef IMILEAGEFILEHANDLER_HPP
#define IMILEAGEFILEHANDLER_HPP

#include <QString>

/*!
 * @brief Interface for reading and writing the mileage of a vehicle to a file.
 * @class IMileageFileHandler
 */
class IMileageFileHandler
{
public:
    virtual ~IMileageFileHandler() = default;
    virtual double readMileage() const = 0;
    virtual void writeMileage(double mileage) const = 0;
};

#endif // IMILEAGEFILEHANDLER_HPP
