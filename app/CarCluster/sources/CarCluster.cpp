#include "CarCluster.h"
#include <QDebug>
#include "DisplayManager.hpp"
#include "MeterController.hpp"
#include "ui_CarCluster.h"

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
    delete this->m_spiController;
    delete this->m_fakeSimulation;
    delete this->ui;
    qDebug() << "CarCluster destroyed";
}

void CarCluster::initializeComponents()
{
    this->m_speedMeterController = new MeterController(20, this);
    this->m_rpmMeterController = new MeterController(2500, this);
    this->m_canBusController = new CanController(this);
    this->m_spiController = new SpiController(this);
    this->m_clusterDisplayManager = new DisplayManager(this,
                                                       this->ui->speedMeterWidget,
                                                       this->m_speedMeterController,
                                                       this->ui->rpmMeterWidget,
                                                       this->m_rpmMeterController,
                                                       this->ui->systemInfoLabel);
}

void CarCluster::configureCanConnection()
{
    qDebug() << "Configuring CAN connection...";
    connect(this->m_canBusController,
            &CanController::speedUpdated,
            this,
            &CarCluster::onSpeedUpdated);
    connect(this->m_canBusController, &CanController::rpmUpdated, this, &CarCluster::onRpmUpdated);

    attemptCanConnection();
}

void CarCluster::onSpeedUpdated(int newSpeed)
{
    qDebug() << "Speed updated:" << newSpeed;
    this->m_speedMeterController->setValue(newSpeed);
}

void CarCluster::onRpmUpdated(int newRpm)
{
    qDebug() << "RPM updated:" << newRpm;
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

void CarCluster::attemptCanConnection()
{
    // if (this->m_canBusController->connectDevice()) {
    if (this->m_spiController->connectDevice()) {
        qDebug() << "CAN connection established!";
    } else {
        qDebug() << "Failed to establish CAN connection. Restarting simulation.";
        // startFakeSimulation();
    }
}
