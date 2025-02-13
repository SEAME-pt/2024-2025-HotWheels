/*!
 * @file IBatteryController.hpp
 * @brief Definition of the IBatteryController interface.
 * @version 0.1
 * @date 2025-02-12
 * @details This file contains the definition of the IBatteryController interface, which
 * is responsible for managing the battery of the vehicle.
 * @author Félix LE BIHAN (@Fle-bihh)
 * @author Tiago Pereira (@t-pereira06)
 * @author Ricardo Melo (@reomelo)
 * @author Michel Batista (@MicchelFAB)
 *
 * @copyright Copyright (c) 2025
 */

#ifndef IBATTERYCONTROLLER_HPP
#define IBATTERYCONTROLLER_HPP

/*!
 * @brief Interface for managing the battery of the vehicle.
 * @class IBatteryController
 */
class IBatteryController
{
public:
	virtual ~IBatteryController() = default;
	virtual float getBatteryPercentage() = 0;
};

#endif // IBATTERYCONTROLLER_HPP
