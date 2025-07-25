#include "I2CController.hpp"
#include <gtest/gtest.h>
#include <gmock/gmock.h>
#include <fcntl.h>
#include <unistd.h>
#include <cstdarg>

// Mock system calls
extern "C" {
	int __real_open(const char *pathname, int flags, ...);
	int __real_ioctl(int fd, unsigned long request, ...);
	ssize_t __real_write(int fd, const void *buf, size_t count);
	ssize_t __real_read(int fd, void *buf, size_t count);

	int __wrap_open(const char *pathname, int flags, ...);
	int __wrap_ioctl(int fd, unsigned long request, ...);
	ssize_t __wrap_write(int fd, const void *buf, size_t count);
	ssize_t __wrap_read(int fd, void *buf, size_t count);
}

// Mocked values
static int mock_fd = 42;
static bool fail_open = false;
static bool fail_ioctl = false;
static bool fail_write = false;
static bool fail_read = false;
static uint8_t fake_read_data[2] = {0xAB, 0xCD};

static bool is_coverage_file(const char *path) {
	return strstr(path, ".gcda") != nullptr || strstr(path, ".gcno") != nullptr;
}

// Wrap open()
int __wrap_open(const char *pathname, int flags, ...) {
	va_list args;
	va_start(args, flags);
	mode_t mode = va_arg(args, mode_t);
	va_end(args);

	if (strstr(pathname, ".gcda") || strstr(pathname, ".gcno") || strstr(pathname, ".gcov")) {
		// Allow coverage tools to write data
		return __real_open(pathname, flags, mode);
	}

	if (!fail_open) {
		return mock_fd;
	} else {
		errno = ENOENT;
		return -1;
	}
}

// Wrap read()
ssize_t __wrap_read(int fd, void *buf, size_t count) {
	if (fail_read || fd != mock_fd) {
		errno = EBADF;
		return -1;
	}
	if (count >= 2) {
		uint8_t *out = static_cast<uint8_t *>(buf);
		out[0] = fake_read_data[0];
		out[1] = fake_read_data[1];
		return 2;
	}
	return 0;
}

// Wrap write()
ssize_t __wrap_write(int fd, const void *buf, size_t count) {
	if (fail_write || fd != mock_fd) {
		errno = EBADF;
		return -1;
	}
	return count;
}

int __wrap_ioctl(int fd, unsigned long request, ...) {
	return fail_ioctl ? -1 : 0;
}

// Test fixture
class I2CControllerTest : public ::testing::Test {
protected:
	void SetUp() override {
		fail_open = false;
		fail_ioctl = false;
		fail_write = false;
		fail_read = false;
		fake_read_data[0] = 0xAB;
		fake_read_data[1] = 0xCD;
	}
};

// Tests

TEST_F(I2CControllerTest, Constructor_SuccessfulOpenAndIoctl) {
	EXPECT_NO_THROW({
		I2CController ctrl("/dev/i2c-1", 0x48);
	});
}

TEST_F(I2CControllerTest, Constructor_OpenFails) {
	fail_open = true;
	EXPECT_NO_THROW({
		I2CController ctrl("/dev/i2c-1", 0x48);
	});  // Should print error but not crash
}

TEST_F(I2CControllerTest, Constructor_IoctlFails) {
	fail_ioctl = true;
	EXPECT_NO_THROW({
		I2CController ctrl("/dev/i2c-1", 0x48);
	});  // Should print error but not crash
}

TEST_F(I2CControllerTest, WriteRegister_Success) {
	I2CController ctrl("/dev/i2c-1", 0x48);
	EXPECT_NO_THROW(ctrl.writeRegister(0x10, 0x1234));
}

TEST_F(I2CControllerTest, WriteRegister_Failure) {
	fail_write = true;
	I2CController ctrl("/dev/i2c-1", 0x48);
	EXPECT_NO_THROW(ctrl.writeRegister(0x10, 0x1234));  // Logs error
}

TEST_F(I2CControllerTest, ReadRegister_Success) {
	I2CController ctrl("/dev/i2c-1", 0x48);
	uint16_t value = ctrl.readRegister(0x10);
	EXPECT_EQ(value, 0xABCD);  // From fake_read_data
}

TEST_F(I2CControllerTest, ReadRegister_FailureWrite) {
	fail_write = true;
	I2CController ctrl("/dev/i2c-1", 0x48);
	uint16_t value = ctrl.readRegister(0x10);  // Logs error
	EXPECT_EQ(value, 0xABCD);  // Still reads old fake data
}

TEST_F(I2CControllerTest, ReadRegister_FailureRead) {
	fail_read = true;
	I2CController ctrl("/dev/i2c-1", 0x48);
	uint16_t value = ctrl.readRegister(0x10);  // Logs error
	EXPECT_EQ(value, 0);
}
