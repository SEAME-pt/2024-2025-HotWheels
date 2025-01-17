#include "CarCluster.h"
#include <QDebug>
#include <QThread>
#include "CanReceiverWorker.hpp"
#include "DisplayManager.hpp"
#include "MeterController.hpp"
#include "SystemInfoUtility.hpp"
#include "ui_CarCluster.h"
#include <unistd.h>

CarCluster::CarCluster(QWidget *parent)
    : QMainWindow(parent)
    , ui(new Ui::CarCluster)
{
    SystemInfoUtility::printClassInfo("CarCluster", SystemInfoUtility::InfoType::CreBegin);
    this->ui->setupUi(this);
    this->initializeComponents();
    this->configureCanConnection();
    SystemInfoUtility::printClassInfo("CarCluster", SystemInfoUtility::InfoType::CreEnd);
}

CarCluster::~CarCluster()
{
    SystemInfoUtility::printClassInfo("CarCluster", SystemInfoUtility::InfoType::DesBegin);
    delete this->m_clusterDisplayManager;
    delete this->m_canBusController;
    delete this->ui;
    delete this->m_speedMeterController;
    delete this->m_rpmMeterController;
    SystemInfoUtility::printClassInfo("CarCluster", SystemInfoUtility::InfoType::DesEnd);
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
    // qDebug() << "Configuring CAN connection...";

    if (attemptCanConnection()) {
        connect(this->m_canBusController, &MCP2515::speedUpdated, this, &CarCluster::onSpeedUpdated);
        connect(this->m_canBusController, &MCP2515::rpmUpdated, this, &CarCluster::onRpmUpdated);
    }
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

bool CarCluster::attemptCanConnection()
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
        qDebug() << "Failed to establish CAN connection.";
        return false;
    }
    return true;
}
