#ifndef METERCONTROLLER_HPP
#define METERCONTROLLER_HPP

#include <QObject>
#include <QTimer>
#include <qqml.h>

class MeterController : public QObject
{
    Q_OBJECT
    Q_PROPERTY(int maxValue READ maxValue CONSTANT)
    Q_PROPERTY(int value READ value WRITE setValue NOTIFY valueChanged)
    QML_ELEMENT

public:
    explicit MeterController(int max_value = 20, QObject *parent = nullptr);
    ~MeterController();

    int value() const;
    int maxValue() const;
    void setValue(int value);
    void setMaxValue(int max_value);

signals:
    void valueChanged();

private:
    int m_value;
    int m_max_value;
};

#endif // METERCONTROLLER_HPP
