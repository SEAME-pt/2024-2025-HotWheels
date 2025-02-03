#ifndef VEHICLEDATAMANAGER_HPP
#define VEHICLEDATAMANAGER_HPP

#include <QObject>
#include <QString>
#include "enums.hpp"

class VehicleDataManager : public QObject
{
    Q_OBJECT

public:
    explicit VehicleDataManager(QObject *parent = nullptr);
    ~VehicleDataManager();

public slots:
    void handleRpmData(int rawRpm);
    void handleSpeedData(float rawSpeed);
    void handleDirectionData(CarDirection rawDirection);
    void handleSteeringData(int rawAngle);
    void handleMileageUpdate(double mileage);

signals:
    void canDataProcessed(float processedSpeed, int processedRpm);
    void engineDataProcessed(CarDirection processedDirection, int processedAngle);
    void mileageUpdated(double mileage);

private:
    float m_speed = 0.0f;
    int m_rpm = 0;
    CarDirection m_carDirection = CarDirection::Stop;
    int m_steeringDirection = 0;
    double m_mileage = 0.0;
    ClusterMetrics m_clusterMetrics = ClusterMetrics::Kilometers;
};

#endif // VEHICLEDATAMANAGER_HPP
