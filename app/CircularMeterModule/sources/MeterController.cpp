#include "MeterController.hpp"
#include <QDebug>
#include <cmath>

MeterController::MeterController(int max_value, QObject *parent)
    : QObject(parent)
    , m_value(0)
    , m_max_value(max_value)
{
    qDebug() << "MeterController created";
}

MeterController::~MeterController()
{
    qDebug() << "MeterController destroyed";
}

int MeterController::value() const
{
    return this->m_value;
}

int MeterController::maxValue() const
{
    return this->m_max_value;
}

void MeterController::setValue(int value)
{
    if (this->m_value == value)
        return;
    this->m_value = value;
    emit this->valueChanged();
}

void MeterController::setMaxValue(int max_value)
{
    if (this->m_max_value == max_value)
        return;

    this->m_max_value = max_value;
}
