#ifndef CIRCULARMETERSETUP_HPP
#define CIRCULARMETERSETUP_HPP

#include <QQuickWidget>
#include <QWidget>
#include "MeterController.hpp"

class CircularMeterSetup
{
public:
    static void setupQuickWidget(QQuickWidget *quickWidget,
                                 QWidget *parentWidget,
                                 MeterController *controller,
                                 const QString &color,
                                 int ticksInterval,
                                 const QString &meterLabel,
                                 int meterFontSize);
};

#endif // CIRCULARMETERSETUP_HPP
