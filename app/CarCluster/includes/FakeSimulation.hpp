#ifndef FAKESIMULATION_HPP
#define FAKESIMULATION_HPP

#include <QFile>
#include <QJsonArray>
#include <QJsonDocument>
#include <QObject>
#include <QTimer>

class FakeSimulation : public QObject
{
    Q_OBJECT

public:
    explicit FakeSimulation(QObject *parent = nullptr);
    ~FakeSimulation();

    void startSimulation();
    void stopSimulation();
    bool isRunning() const;

signals:
    void speedUpdated(int newSpeed);
    void rpmUpdated(int newRpm);
    void simulationFinished();

private slots:
    void sendFakeUpdates();

private:
    void loadSimulationData(const QString &fileName, QList<int> &targetList);

    QTimer *simulationTimer;
    QList<int> fakeSpeeds;
    QList<int> fakeRpms;
    int fakeDataIndex;
    bool simulationActive;
};

#endif // FAKESIMULATION_HPP
