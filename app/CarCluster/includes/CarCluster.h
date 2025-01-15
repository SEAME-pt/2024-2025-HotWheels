#ifndef CARCLUSTER_H
#define CARCLUSTER_H

#include <QMainWindow>
#include "FakeSimulation.hpp"
#include "MCP2515.hpp"

class QQuickView;
class MeterController;
class DisplayManager;

QT_BEGIN_NAMESPACE
namespace Ui {
class CarCluster;
}
QT_END_NAMESPACE

class CarCluster : public QMainWindow
{
    Q_OBJECT

public:
    explicit CarCluster(QWidget *parent = nullptr);
    ~CarCluster();

private:
    Ui::CarCluster *ui;

    MeterController *m_speedMeterController;
    MeterController *m_rpmMeterController;
    DisplayManager *m_clusterDisplayManager;
    MCP2515 *m_canBusController;
    FakeSimulation *m_fakeSimulation;

    void initializeComponents();
    void configureCanConnection();
    void attemptCanConnection();
    void startFakeSimulation();

private slots:
    void onSpeedUpdated(int newSpeed);
    void onRpmUpdated(int newRpm);
    void onSimulationFinished();
};

#endif // CARCLUSTER_H
