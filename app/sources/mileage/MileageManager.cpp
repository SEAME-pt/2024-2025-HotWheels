#include "MileageManager.hpp"
#include <QDebug>
#include "MileageCalculator.hpp"
#include "MileageFileHandler.hpp"

MileageManager::MileageManager(const QString &filePath,
                               IMileageCalculator *calculator,
                               IMileageFileHandler *fileHandler,
                               QObject *parent)
    : QObject(parent)
    , m_calculator(calculator ? calculator : new MileageCalculator())
    , m_fileHandler(fileHandler ? fileHandler : new MileageFileHandler(filePath))
    , m_ownCalculator(calculator == nullptr)
    , m_ownFileHandler(fileHandler == nullptr)
    , m_totalMileage(0.0)
{}

MileageManager::~MileageManager()
{
    shutdown();

    // Only delete instances if they were created internally
    if (m_ownCalculator) {
        delete m_calculator;
    }
    if (m_ownFileHandler) {
        delete m_fileHandler;
    }
}

void MileageManager::initialize()
{
    m_totalMileage = m_fileHandler->readMileage();

    connect(&m_updateTimer, &QTimer::timeout, this, &MileageManager::updateMileage);
    m_updateTimer.start(1000);

    connect(&m_persistenceTimer, &QTimer::timeout, this, &MileageManager::saveMileage);
    m_persistenceTimer.start(10000);
}

void MileageManager::shutdown()
{
    saveMileage();
    m_updateTimer.stop();
    m_persistenceTimer.stop();
}

void MileageManager::onSpeedUpdated(float speed)
{
    m_calculator->addSpeed(speed);
}

void MileageManager::updateMileage()
{
    double distance = m_calculator->calculateDistance();
    m_totalMileage += distance;
    emit mileageUpdated(m_totalMileage);
}

void MileageManager::saveMileage()
{
    m_fileHandler->writeMileage(m_totalMileage);
}
