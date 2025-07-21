// test_SystemCommandExecutor.cpp

#include "SystemCommandExecutor.hpp"
#include <gtest/gtest.h>
#include <QTemporaryFile>
#include <QTextStream>

class SystemCommandExecutorTest : public ::testing::Test {
protected:
	SystemCommandExecutor executor;
};

TEST_F(SystemCommandExecutorTest, ExecuteCommand_ReturnsExpectedOutput)
{
	QString result = executor.executeCommand("echo HelloWorld");
	EXPECT_EQ(result, "HelloWorld");
}

TEST_F(SystemCommandExecutorTest, ExecuteCommand_ReturnsEmptyOnInvalidCommand)
{
	QString result = executor.executeCommand("nonexistentcommand_hopefully");
	EXPECT_TRUE(result.isEmpty());
}

TEST_F(SystemCommandExecutorTest, ReadFile_ReturnsFileContents)
{
	QTemporaryFile tempFile;
	ASSERT_TRUE(tempFile.open());
	QTextStream out(&tempFile);
	out << "TestContent\nSecondLine";
	out.flush();

	QString output = executor.readFile(tempFile.fileName());
	EXPECT_EQ(output, "TestContent");
}

TEST_F(SystemCommandExecutorTest, ReadFile_ReturnsEmptyOnMissingFile)
{
	QString output = executor.readFile("/path/that/does/not/exist.txt");
	EXPECT_TRUE(output.isEmpty());
}
