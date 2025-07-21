#include <gtest/gtest.h>
#include <QFile>
#include <QTemporaryDir>
#include <QTemporaryFile>
#include <QString>
#include "FileController.hpp"

using namespace FileController;

class FileControllerTest : public ::testing::Test {
protected:
	QTemporaryDir tempDir;

	QString tempFilePath(const QString& filename) {
		return tempDir.path() + "/" + filename;
	}
};

TEST_F(FileControllerTest, OpenFileSuccess) {
	QString path = tempFilePath("testOpen.txt");
	QFile file(path);

	ASSERT_TRUE(open(file, QIODevice::WriteOnly));
	file.close();
}

TEST_F(FileControllerTest, WriteToFile) {
	QString path = tempFilePath("testWrite.txt");
	QFile file(path);
	ASSERT_TRUE(open(file, QIODevice::WriteOnly | QIODevice::Text));
	ASSERT_TRUE(write(file, "Hello, world!"));
	file.close();

	// Re-open and read back
	QFile fileRead(path);
	ASSERT_TRUE(open(fileRead, QIODevice::ReadOnly | QIODevice::Text));
	QString content = read(fileRead);
	fileRead.close();
	ASSERT_EQ(content, "Hello, world!");
}

TEST_F(FileControllerTest, ReadLineFromFile) {
	QString path = tempFilePath("testRead.txt");
	QFile file(path);
	ASSERT_TRUE(open(file, QIODevice::WriteOnly | QIODevice::Text));
	write(file, "Line 1");
	write(file, "Line 2");
	file.close();

	QFile fileRead(path);
	ASSERT_TRUE(open(fileRead, QIODevice::ReadOnly | QIODevice::Text));
	QString firstLine = read(fileRead);
	fileRead.close();

	ASSERT_EQ(firstLine, "Line 1");
}

TEST_F(FileControllerTest, FileExists) {
	QString path = tempFilePath("testExist.txt");
	QFile file(path);
	ASSERT_TRUE(open(file, QIODevice::WriteOnly));
	file.close();

	ASSERT_TRUE(exists(path));
	ASSERT_FALSE(exists(tempFilePath("doesNotExist.txt")));
}

TEST_F(FileControllerTest, OpenInvalidFileFails) {
	QFile file("/nonexistent/path/file.txt");
	ASSERT_FALSE(open(file, QIODevice::ReadOnly));
}
