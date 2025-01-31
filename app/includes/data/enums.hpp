/**
 * @file enums.hpp
 * @brief 
 * @version 0.1
 * @date 2025-01-31
 * @details
 * @note 
 * @author Félix LE BIHAN (@Fle-bihh)
 * @author Tiago Pereira (@t-pereira06)
 * @author Ricardo Melo (@reomelo)
 * @author Michel Batista (@MicchelFAB)
 *
 * @copyright Copyright (c) 2025
 */

#ifndef ENUMS_HPP
#define ENUMS_HPP

#include <QtCore/qmetatype.h>

enum class ComponentStatus { Idle, Starting, Operational, Down };
enum class DrivingMode { Manual, Automatic };
enum class ClusterTheme { Dark, Light };
enum class ClusterMetrics { Miles, Kilometers };
enum class CarDirection { Drive, Reverse, Stop };

Q_DECLARE_METATYPE(ComponentStatus)
Q_DECLARE_METATYPE(DrivingMode)
Q_DECLARE_METATYPE(ClusterTheme)
Q_DECLARE_METATYPE(ClusterMetrics)
Q_DECLARE_METATYPE(CarDirection)

#endif // ENUMS_HPP
