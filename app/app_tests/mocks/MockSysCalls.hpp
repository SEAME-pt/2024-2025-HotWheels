/**
 * @file MockSysCalls.hpp
 * @author Félix LE BIHAN (@Fle-bihh)
 * @author Ricardo Melo (@reomelo)
 * @author Tiago Pereira (@t-pereira06)
 * @author Michel Batista (@MicchelFAB)
 * @brief File containing Mock classes to test the system calls.
 * @version 0.1
 * @date 2025-01-30
 * 
 * @copyright Copyright (c) 2025
 * 
 */

#ifndef MOCKSYSCALLS_HPP
#define MOCKSYSCALLS_HPP

#include <fcntl.h>
#include <gmock/gmock.h>
#include <sys/ioctl.h>
#include <unistd.h>

/**
 * @class MockSysCalls
 * @brief Class to emulate the behavior of the system calls.
 */
class MockSysCalls
{
public:
    /**
     * @brief Get the instance object
     * @return MockSysCalls& 
     */
    static MockSysCalls &instance()
    {
        static MockSysCalls instance;
        return instance;
    }

    MOCK_METHOD(int, open, (const char *path, int flags), ()); /// Mocked method to open a file.
    MOCK_METHOD(int, ioctl, (int fd, unsigned long request), ()); /// Mocked method to perform an I/O control operation.
    MOCK_METHOD(int, close, (int fd), ()); /// Mocked method to close a file.

private:
    MockSysCalls() = default; /// Constructor of the class set as default.
    ~MockSysCalls() = default; /// Destructor of the class set as default.
    MockSysCalls(const MockSysCalls &) = delete; /// Copy constructor of the class set as delete.
    MockSysCalls &operator=(const MockSysCalls &) = delete; /// Operator of the class set as delete.
};

/**
 * @brief Mocked open function.
 * @param path 
 * @param flags 
 * @return a mocked file descriptor.
 * @retval int
 */
inline int mock_open(const char *path, int flags, ...)
{
    return MockSysCalls::instance().open(path, flags);
}

/**
 * @brief Mocked ioctl function.
 * @param fd 
 * @param request 
 * @return a mocked ioctl return value.
 * @retval int
 */
inline int mock_ioctl(int fd, unsigned long request, ...)
{
    return MockSysCalls::instance().ioctl(fd, request);
}

/**
 * @brief Mocked close function.
 * @param fd 
 * @return a mocked close return value.
 * @retval int
 */
inline int mock_close(int fd)
{
    return MockSysCalls::instance().close(fd);
}

#endif // MOCKSYSCALLS_HPP
