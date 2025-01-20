#ifndef CARMANAGER_H
#define CARMANAGER_H

#include <QMainWindow>
#include "CanBusManager.hpp"
#include "ControlsManager.hpp"
#include "DataManager.hpp"
#include "DisplayManager.hpp"
#include "SystemManager.hpp"

QT_BEGIN_NAMESPACE
namespace Ui {
class CarManager;
}
QT_END_NAMESPACE

class CarManager : public QMainWindow
{
    Q_OBJECT

public:
    CarManager(QWidget *parent = nullptr);
    ~CarManager();

private:
    Ui::CarManager *ui;
    DataManager *m_dataManager;
    CanBusManager *m_canBusManager;
    ControlsManager *m_controlsManager;
    DisplayManager *m_displayManager;
    SystemManager *m_systemManager;

    void initializeComponents();
    void initializeDataManager();
    void initializeCanBusManager();
    void initializeControlsManager();
    void initializeDisplayManager();
    void initializeSystemManager();
};

#endif // CARMANAGER_H
