#include "MileageManager.hpp"
#include <QDebug>

MileageManager::MileageManager(const QString &filePath, QObject *parent)
    : QObject(parent)
    , fileHandler(filePath)
    , totalMileage(0.0)
{}

MileageManager::~MileageManager()
{
    shutdown();
}

void MileageManager::initialize()
{
    // Load initial mileage from file
    totalMileage = fileHandler.readMileage();

    // Configure update timer (every 5 seconds)
    connect(&updateTimer, &QTimer::timeout, this, &MileageManager::updateMileage);
    updateTimer.start(1000);

    // Configure persistence timer (every 10 seconds)
    connect(&persistenceTimer, &QTimer::timeout, this, &MileageManager::saveMileage);
    persistenceTimer.start(10000);
}

void MileageManager::shutdown()
{
    saveMileage(); // Ensure mileage is saved on shutdown
    updateTimer.stop();
    persistenceTimer.stop();
}

void MileageManager::onSpeedUpdated(float speed)
{
    calculator.addSpeed(speed);
}

void MileageManager::updateMileage()
{
    // Calculate distance for the last interval
    // qDebug() << "Updating mileage";
    double distance = calculator.calculateDistance();
    totalMileage += distance;

    // Emit updated mileage
    // qDebug() << "Updating mileage" << totalMileage;
    emit mileageUpdated(totalMileage);
}

void MileageManager::saveMileage()
{
    fileHandler.writeMileage(totalMileage);
}
