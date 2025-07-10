#include <gtest/gtest.h>
#include <QApplication>

int main(int argc, char** argv) {
	QApplication app(argc, argv);  // Only ONE instance here
	::testing::InitGoogleTest(&argc, argv);
	return RUN_ALL_TESTS();
}
