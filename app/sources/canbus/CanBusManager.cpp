/**
 * @file CanBusManager.cpp
 * @author Michel Batista (michel_fab@outlook.com)
 * @author 
 * @brief Implementation of the CanBusManager class.
 * @version 0.1
 * @date 2025-01-29
 * 
 * @copyright Copyright (c) 2025
 * 
 */

#include "CanBusManager.hpp"
#include <QDebug>
#include "MCP2515Controller.hpp"

CanBusManager::CanBusManager(const std::string &spi_device, QObject *parent)
    : QObject(parent)
{
    m_controller = new MCP2515Controller(spi_device);
    ownsMCP2515Controller = true;
    connectSignals();
}

CanBusManager::CanBusManager(IMCP2515Controller *controller, QObject *parent)
    : QObject(parent)
    , m_controller(controller)
{
    ownsMCP2515Controller = false;
    connectSignals();
}

CanBusManager::~CanBusManager()
{
    if (m_thread) {
        m_controller->stopReading();
        m_thread->disconnect();
        m_thread->quit();
        m_thread->wait();

        delete m_thread;
    }

    if (ownsMCP2515Controller) {
        delete m_controller;
    }
}

void CanBusManager::connectSignals()
{
    connect(m_controller, &IMCP2515Controller::speedUpdated, this, &CanBusManager::speedUpdated);
    connect(m_controller, &IMCP2515Controller::rpmUpdated, this, &CanBusManager::rpmUpdated);
}

bool CanBusManager::initialize()
{
    if (!m_controller->init()) {
        return false;
    }

    m_thread = new QThread(this);
    m_controller->moveToThread(m_thread);

    connect(m_thread, &QThread::started, m_controller, &IMCP2515Controller::processReading);
    connect(m_thread, &QThread::finished, m_controller, &QObject::deleteLater);
    connect(m_thread, &QThread::finished, m_thread, &QObject::deleteLater);

    m_thread->start();
    return true;
}
