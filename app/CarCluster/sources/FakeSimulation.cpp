#include "FakeSimulation.hpp"
#include <QDebug>
#include <QDir>

FakeSimulation::FakeSimulation(QObject *parent)
    : QObject(parent)
    , simulationTimer(new QTimer(this))
    , fakeDataIndex(0)
    , simulationActive(false)
{
    connect(simulationTimer, &QTimer::timeout, this, &FakeSimulation::sendFakeUpdates);
}

FakeSimulation::~FakeSimulation()
{
    stopSimulation();
}

void FakeSimulation::startSimulation()
{
    if (simulationActive)
        return;

    simulationActive = true;
    fakeDataIndex = 0;

    // Load simulation data
    fakeSpeeds.clear();
    fakeRpms.clear();
    loadSimulationData(":/data/speed-simulation.fake.json", fakeSpeeds);
    loadSimulationData(":/data/rpm-simulation.fake.json", fakeRpms);

    simulationTimer->start(250);
}

void FakeSimulation::stopSimulation()
{
    if (!simulationActive)
        return;

    simulationTimer->stop();
    simulationActive = false;
    emit simulationFinished();
}

bool FakeSimulation::isRunning() const
{
    return simulationActive;
}

void FakeSimulation::sendFakeUpdates()
{
    if (fakeDataIndex < fakeSpeeds.size() && fakeDataIndex < fakeRpms.size()) {
        emit speedUpdated(fakeSpeeds.at(fakeDataIndex));
        emit rpmUpdated(fakeRpms.at(fakeDataIndex));
        fakeDataIndex++;
    } else {
        stopSimulation();
    }
}

void FakeSimulation::loadSimulationData(const QString &fileName, QList<int> &targetList)
{
    QFile file(fileName);

    if (!file.open(QIODevice::ReadOnly)) {
        qDebug() << "Failed to open simulation data file:" << fileName;
        return;
    }

    QByteArray fileContents = file.readAll();
    fileContents = fileContents.trimmed();

    QJsonDocument doc = QJsonDocument::fromJson(fileContents);
    if (!doc.isArray()) {
        qDebug() << "Invalid JSON format in file:" << fileName;
        return;
    }

    QJsonArray jsonArray = doc.array();
    for (const QJsonValue &value : jsonArray) {
        targetList.append(value.toInt());
    }
}
