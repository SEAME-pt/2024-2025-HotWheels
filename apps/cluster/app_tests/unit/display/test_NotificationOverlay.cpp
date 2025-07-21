#include <gtest/gtest.h>
#include <QApplication>
#include <QTest>
#include <QSignalSpy>
#include "NotificationOverlay.hpp"

// Required to initialize Qt GUI application
int argc = 0;
char** argv = nullptr;

class NotificationOverlayTest : public ::testing::Test {
protected:
	QWidget parent;
	NotificationOverlay* overlay;

	void SetUp() override {
		overlay = new NotificationOverlay(&parent);
		overlay->resize(400, 300); // Simulate a real window size
	}

	void TearDown() override {
		delete overlay;
	}
};

TEST_F(NotificationOverlayTest, ShowsNotificationImmediately) {
	overlay->showNotification("Test Message", NotificationLevel::Info, 1000);

	EXPECT_TRUE(overlay->isVisible());
}

TEST_F(NotificationOverlayTest, PersistsWhenDurationIsZero) {
	overlay->showNotification("Persistent", NotificationLevel::Warning, 0);

	// Wait for 1.5 seconds to simulate idle time
	QTest::qWait(1500);

	EXPECT_TRUE(overlay->isVisible());  // Still visible
}

TEST_F(NotificationOverlayTest, HidesAfterDuration) {
	overlay->showNotification("Timed", NotificationLevel::Info, 500);

	QTest::qWait(1100);  // Wait longer than fade + duration

	EXPECT_FALSE(overlay->isVisible());
}

TEST_F(NotificationOverlayTest, ManualHidePersistent) {
	overlay->showNotification("Manual Hide", NotificationLevel::Warning, 0);  // Persistent

	EXPECT_TRUE(overlay->isVisible());

	overlay->hideNotification();  // Should start fade out
	QTest::qWait(600);  // Wait for fade out to complete

	EXPECT_FALSE(overlay->isVisible());
}
