#ifndef MOCKMILEAGEFILEHANDLER_HPP
#define MOCKMILEAGEFILEHANDLER_HPP

#include "IMileageFileHandler.hpp"
#include <gmock/gmock.h>

class MockMileageFileHandler : public IMileageFileHandler
{
public:
    MOCK_METHOD(double, readMileage, (), (const, override));
    MOCK_METHOD(void, writeMileage, (double mileage), (const, override));
};

#endif // MOCKMILEAGEFILEHANDLER_HPP
