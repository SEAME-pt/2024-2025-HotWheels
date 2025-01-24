#include "CanBusManager.hpp"
#include <QDebug>

CanBusManager::CanBusManager(const std::string &spi_device, QObject *parent)
    : QObject(parent)
{
    m_controller = new MCP2515Controller(spi_device);

    connect(m_controller, &MCP2515Controller::speedUpdated, this, &CanBusManager::speedUpdated);
    connect(m_controller, &MCP2515Controller::rpmUpdated, this, &CanBusManager::rpmUpdated);
}

CanBusManager::~CanBusManager()
{
    if (m_thread) {
        m_controller->stopReading();

        m_thread->disconnect();

        m_thread->quit();
        m_thread->wait();
    }

    delete m_controller;
    delete m_thread;
}

bool CanBusManager::initialize()
{
    if (!m_controller->init()) {
        qDebug() << "Failed to initialize MCP2515.";
        return false;
    }

    m_thread = new QThread(this);
    m_controller->moveToThread(m_thread);

    connect(m_thread, &QThread::started, m_controller, &MCP2515Controller::processReading);
    connect(m_thread, &QThread::finished, m_controller, &QObject::deleteLater);
    connect(m_thread, &QThread::finished, m_thread, &QObject::deleteLater);

    m_thread->start();
    return true;
}
