#ifndef MILEAGEFILEHANDLER_HPP
#define MILEAGEFILEHANDLER_HPP

#include <QString>

class MileageFileHandler
{
public:
    explicit MileageFileHandler(const QString &filePath);
    ~MileageFileHandler() = default;

    double readMileage() const;
    void writeMileage(double mileage) const;
    void ensureFileExists() const;

private:
    QString filePath;
};

#endif // MILEAGEFILEHANDLER_HPP
