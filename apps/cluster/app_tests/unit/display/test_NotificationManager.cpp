#include <gtest/gtest.h>
#include <QApplication>
#include <QElapsedTimer>
#include <QTest>
#include "NotificationManager.hpp"
#include "NotificationOverlay.hpp"

class NotificationManagerIntegrationTest : public ::testing::Test {
protected:
	QWidget parent;
	NotificationOverlay* overlay = nullptr;
	NotificationManager* manager = nullptr;

	void SetUp() override {
		overlay = new NotificationOverlay(&parent);
		overlay->resize(400, 300);
		manager = NotificationManager::instance();
		manager->initialize(overlay);
	}

	void TearDown() override {
		manager->initialize(nullptr);  // Avoid dangling pointer
		delete overlay;
	}
};

TEST_F(NotificationManagerIntegrationTest, ShowsNotification) {
	manager->enqueueNotification("Hello", NotificationLevel::Info, 500);
	EXPECT_TRUE(overlay->isVisible());
}

TEST_F(NotificationManagerIntegrationTest, HidesNotificationAfterTimeout) {
	manager->enqueueNotification("Auto-hide", NotificationLevel::Info, 300);

	QElapsedTimer timer;
	timer.start();
	while (timer.elapsed() < 1000) {
		QCoreApplication::processEvents(QEventLoop::AllEvents, 50);
		QTest::qWait(10);
	}

	EXPECT_FALSE(overlay->isVisible());
}

TEST_F(NotificationManagerIntegrationTest, PersistentNotificationStays) {
	manager->showPersistentNotification("Stay!", NotificationLevel::Warning);

	QTest::qWait(1000);
	EXPECT_TRUE(overlay->isVisible());
}

TEST_F(NotificationManagerIntegrationTest, clearNotificationHidesPersistent) {
	manager->showPersistentNotification("Remove Me", NotificationLevel::Warning);
	EXPECT_TRUE(overlay->isVisible());

	manager->clearNotification();
	QTest::qWait(600);  // Fade-out time is 500ms
	EXPECT_FALSE(overlay->isVisible());
}
