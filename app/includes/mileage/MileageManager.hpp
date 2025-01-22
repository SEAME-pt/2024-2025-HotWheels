#ifndef MILEAGEMANAGER_HPP
#define MILEAGEMANAGER_HPP

#include <QObject>
#include <QTimer>
#include "MileageCalculator.hpp"
#include "MileageFileHandler.hpp"

class MileageManager : public QObject
{
    Q_OBJECT

public:
    explicit MileageManager(const QString &filePath, QObject *parent = nullptr);
    ~MileageManager();

    void initialize();
    void shutdown();

public slots:
    void onSpeedUpdated(float speed);
    void updateMileage();
    void saveMileage();

signals:
    void mileageUpdated(double mileage);

private:
    MileageCalculator calculator;
    MileageFileHandler fileHandler;
    QTimer updateTimer;      // Updates mileage every 5 seconds
    QTimer persistenceTimer; // Saves mileage periodically
    double totalMileage;
};

#endif // MILEAGEMANAGER_HPP
