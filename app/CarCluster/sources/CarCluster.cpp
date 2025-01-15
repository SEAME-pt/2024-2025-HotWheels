#include "CarCluster.h"
#include <QDebug>
#include <QThread>
#include "CanReceiverWorker.hpp"
#include "DisplayManager.hpp"
#include "MeterController.hpp"
#include "ui_CarCluster.h"
#include <unistd.h>

CarCluster::CarCluster(QWidget *parent)
    : QMainWindow(parent)
    , ui(new Ui::CarCluster)
    , m_fakeSimulation(new FakeSimulation(this))
{
    qDebug() << "CarCluster is being created";
    this->ui->setupUi(this);
    this->initializeComponents();
    this->configureCanConnection();
    qDebug() << "CarCluster created";
}

CarCluster::~CarCluster()
{
    qDebug() << "CarCluster is being destroyed";
    delete this->m_clusterDisplayManager;
    delete this->m_canBusController;
    delete this->m_fakeSimulation;
    delete this->ui;
    qDebug() << "CarCluster destroyed";
}

void CarCluster::initializeComponents()
{
    this->m_speedMeterController = new MeterController(10, this);
    this->m_rpmMeterController = new MeterController(1700, this);
    this->m_clusterDisplayManager = new DisplayManager(this,
                                                       this->ui->speedMeterWidget,
                                                       this->m_speedMeterController,
                                                       this->ui->rpmMeterWidget,
                                                       this->m_rpmMeterController,
                                                       this->ui->systemInfoLabel);
    this->m_canBusController = new MCP2515("/dev/spidev0.0");
}

void CarCluster::configureCanConnection()
{
    qDebug() << "Configuring CAN connection...";
    connect(this->m_canBusController, &MCP2515::speedUpdated, this, &CarCluster::onSpeedUpdated);
    connect(this->m_canBusController, &MCP2515::rpmUpdated, this, &CarCluster::onRpmUpdated);

    attemptCanConnection();
}

void CarCluster::onSpeedUpdated(int newSpeed)
{
    // qDebug() << "Speed updated:" << newSpeed;
    this->m_speedMeterController->setValue(newSpeed);
}

void CarCluster::onRpmUpdated(int newRpm)
{
    // qDebug() << "RPM updated:" << newRpm;
    this->m_rpmMeterController->setValue(newRpm);
}

void CarCluster::startFakeSimulation()
{
    qDebug() << "Starting fake simulation...";
    m_fakeSimulation->startSimulation();
    connect(m_fakeSimulation, &FakeSimulation::speedUpdated, this, &CarCluster::onSpeedUpdated);
    connect(m_fakeSimulation, &FakeSimulation::rpmUpdated, this, &CarCluster::onRpmUpdated);
    connect(m_fakeSimulation,
            &FakeSimulation::simulationFinished,
            this,
            &CarCluster::onSimulationFinished);
}

void CarCluster::onSimulationFinished()
{
    qDebug() << "Simulation finished. Retrying CAN connection...";
    attemptCanConnection();
}

// void CarCluster::attemptCanConnection()
// {
//     if (this->m_canBusController->init()) {
//         qDebug() << "CAN connection established! Starting to read.";

//         // while (1) {
//         //     this->m_canBusController->receive();
//         //     usleep(10000);
//         // }
//         // // Set up a QTimer to call receive periodically
//         // QTimer *canReadTimer = new QTimer(this);
//         // connect(canReadTimer, &QTimer::timeout, this->m_canBusController, &MCP2515::receive);
//         // canReadTimer->start(10); // Trigger every 10 ms
//     } else {
//         qDebug() << "Failed to establish CAN connection. Restarting simulation.";
//         startFakeSimulation();
//     }
// }

void CarCluster::attemptCanConnection()
{
    if (this->m_canBusController->init()) {
        qDebug() << "CAN connection established! Starting to read.";

        QThread *thread = new QThread(this);
        CanReceiverWorker *worker = new CanReceiverWorker(this->m_canBusController);

        worker->moveToThread(thread);

        connect(thread, &QThread::started, worker, &CanReceiverWorker::process);
        connect(thread, &QThread::finished, worker, &QObject::deleteLater);
        connect(thread, &QThread::finished, thread, &QObject::deleteLater);

        thread->start();
    } else {
        qDebug() << "Failed to establish CAN connection. Restarting simulation.";
        startFakeSimulation();
    }
}
