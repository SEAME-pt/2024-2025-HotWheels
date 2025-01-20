#ifndef CONTROLSMANAGER_HPP
#define CONTROLSMANAGER_HPP

#include <QObject>
#include <QThread>
#include "EngineController.hpp"
#include "JoysticksController.hpp"

class ControlsManager : public QObject
{
    Q_OBJECT

private:
    EngineController m_engineController;
    JoysticksController *m_manualController;
    QThread *m_manualControllerThread;
    DrivingMode m_currentMode;

public:
    explicit ControlsManager(QObject *parent = nullptr);
    ~ControlsManager();

    void setMode(DrivingMode mode);

public slots:
    void drivingModeUpdated(DrivingMode newMode);

signals:
    void directionChanged(CarDirection newDirection);
    void steeringChanged(int newAngle);
};

#endif // CONTROLSMANAGER_HPP
