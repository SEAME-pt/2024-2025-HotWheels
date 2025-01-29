#ifndef MOCKSYSCALLS_HPP
#define MOCKSYSCALLS_HPP

#include <fcntl.h>
#include <gmock/gmock.h>
#include <sys/ioctl.h>
#include <unistd.h>

class MockSysCalls
{
public:
    static MockSysCalls &instance()
    {
        static MockSysCalls instance;
        return instance;
    }

    MOCK_METHOD(int, open, (const char *path, int flags), ());
    MOCK_METHOD(int, ioctl, (int fd, unsigned long request), ());
    MOCK_METHOD(int, close, (int fd), ());

private:
    MockSysCalls() = default;
    ~MockSysCalls() = default;
    MockSysCalls(const MockSysCalls &) = delete;
    MockSysCalls &operator=(const MockSysCalls &) = delete;
};

inline int mock_open(const char *path, int flags, ...)
{
    return MockSysCalls::instance().open(path, flags);
}

inline int mock_ioctl(int fd, unsigned long request, ...)
{
    return MockSysCalls::instance().ioctl(fd, request);
}

inline int mock_close(int fd)
{
    return MockSysCalls::instance().close(fd);
}

#endif // MOCKSYSCALLS_HPP
