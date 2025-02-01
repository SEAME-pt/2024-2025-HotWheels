/**
 * @file MockSysCalls.hpp
 * @brief File containing Mock classes to test the system calls.
 *
 * This file provides a mock implementation of system calls for testing
 * purposes. It uses Google Mock to create mock methods for open, ioctl, and
 * close system calls.
 *
 * @version 0.1
 * @date 2025-01-30
 * @author Félix LE BIHAN (@Fle-bihh)
 * @author Ricardo Melo (@reomelo)
 * @author Tiago Pereira (@t-pereira06)
 * @author Michel Batista (@MicchelFAB)
 *
 * @section License
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
class MockSysCalls {
public:
  /**
   * @brief Get the instance object
   * @return MockSysCalls&
   */
  static MockSysCalls &instance() {
    static MockSysCalls instance;
    return instance;
  }

  /** @brief Mocked method to open a file. */
  MOCK_METHOD(int, open, (const char *path, int flags), ());
  /** @brief Mocked method to perform an I/O control operation. */
  MOCK_METHOD(int, ioctl, (int fd, unsigned long request), ());
  /** @brief Mocked method to close a file. */
  MOCK_METHOD(int, close, (int fd), ());

private:
  /** @brief Constructor of the class set as default. */
  MockSysCalls() = default;
  /** @brief Destructor of the class set as default. */
  ~MockSysCalls() = default;
  /** @brief Copy constructor of the class set as delete. */
  MockSysCalls(const MockSysCalls &) = delete;
  /** @brief Operator of the class set as delete. */
  MockSysCalls &operator=(const MockSysCalls &) = delete;
};

/**
 * @brief Mocked open function.
 * @param path The path to the file to open.
 * @param flags The flags for opening the file.
 * @return a mocked file descriptor.
 * @retval int
 */
inline int mock_open(const char *path, int flags, ...) {
  return MockSysCalls::instance().open(path, flags);
}

/**
 * @brief Mocked ioctl function.
 * @param fd The file descriptor.
 * @param request The request code.
 * @return a mocked ioctl return value.
 * @retval int
 */
inline int mock_ioctl(int fd, unsigned long request, ...) {
  return MockSysCalls::instance().ioctl(fd, request);
}

/**
 * @brief Mocked close function.
 * @param fd The file descriptor.
 * @return a mocked close return value.
 * @retval int
 */
inline int mock_close(int fd) { return MockSysCalls::instance().close(fd); }

#endif // MOCKSYSCALLS_HPP
