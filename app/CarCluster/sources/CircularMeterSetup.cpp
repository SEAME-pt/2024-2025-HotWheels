#include "CircularMeterSetup.hpp"
#include <QQmlComponent>
#include <QQmlContext>
#include <QVBoxLayout>
#include "qqmlengine.h"

void CircularMeterSetup::setupQuickWidget(QQuickWidget *quickWidget,
                                          QWidget *parentWidget,
                                          MeterController *controller,
                                          const QString &color,
                                          int ticksInterval,
                                          const QString &meterLabel,
                                          int meterFontSize)
{
    QQmlEngine *engine = quickWidget->engine();
    QQmlComponent component(engine, QUrl("qrc:/CircularMeterModule/CircularMeter.qml"));
    QQmlContext *context = new QQmlContext(engine->rootContext(), quickWidget);
    context->setContextProperty("meterController", controller);
    context->setContextProperty("parentColor", color);
    context->setContextProperty("ticksInterval", ticksInterval);
    context->setContextProperty("meterLabel", meterLabel);
    context->setContextProperty("meterFontSize", meterFontSize);

    QObject *circularMeterObject = component.create(context);
    quickWidget->setContent(QUrl("qrc:/CircularMeterModule/CircularMeter.qml"),
                            &component,
                            circularMeterObject);

    quickWidget->setResizeMode(QQuickWidget::SizeRootObjectToView);
    QVBoxLayout *layout = new QVBoxLayout(parentWidget);
    layout->addWidget(quickWidget);
}
