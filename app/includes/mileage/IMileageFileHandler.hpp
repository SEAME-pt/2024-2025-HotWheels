#ifndef IMILEAGEFILEHANDLER_HPP
#define IMILEAGEFILEHANDLER_HPP

#include <QString>

class IMileageFileHandler
{
public:
    virtual ~IMileageFileHandler() = default;
    virtual double readMileage() const = 0;
    virtual void writeMileage(double mileage) const = 0;
};

#endif // IMILEAGEFILEHANDLER_HPP
