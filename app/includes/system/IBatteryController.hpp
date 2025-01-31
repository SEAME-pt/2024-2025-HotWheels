#ifndef IBATTERYCONTROLLER_HPP
#define IBATTERYCONTROLLER_HPP

class IBatteryController
{
public:
    virtual ~IBatteryController() = default;
    virtual float getBatteryPercentage() = 0;
};

#endif // IBATTERYCONTROLLER_HPP
