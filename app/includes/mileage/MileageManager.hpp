#ifndef MILEAGEMANAGER_HPP
#define MILEAGEMANAGER_HPP

#include <QObject>
#include <QTimer>
#include "IMileageCalculator.hpp"
#include "IMileageFileHandler.hpp"

class MileageManager : public QObject
{
    Q_OBJECT

public:
    explicit MileageManager(const QString &filePath,
                            IMileageCalculator *calculator = nullptr,
                            IMileageFileHandler *fileHandler = nullptr,
                            QObject *parent = nullptr);
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
    IMileageCalculator *m_calculator;
    IMileageFileHandler *m_fileHandler;
    bool m_ownCalculator;
    bool m_ownFileHandler;
    QTimer m_updateTimer;
    QTimer m_persistenceTimer;
    double m_totalMileage;
};

#endif // MILEAGEMANAGER_HPP
