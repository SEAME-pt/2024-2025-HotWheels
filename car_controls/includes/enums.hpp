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
