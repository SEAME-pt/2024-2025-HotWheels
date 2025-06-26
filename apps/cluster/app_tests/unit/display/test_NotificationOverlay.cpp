#include <QApplication>
#include <QTest>
#include <gtest/gtest.h>
#include "NotificationOverlay.hpp"

class NotificationOverlayTest : public ::testing::Test {
protected:
	QApplication* app;
	QWidget* parentWidget;
	NotificationOverlay* overlay;

	void SetUp() override {
		int argc = 0;
		app = new QApplication(argc, nullptr);  // Needed for widgets
		parentWidget = new QWidget();
		parentWidget->resize(800, 600);
		overlay = new NotificationOverlay(parentWidget);
		parentWidget->show();
	}

	void TearDown() override {
		delete overlay;
		delete parentWidget;
		delete app;
	}
};

/*!
 * @test Tests the showNotification method with Info level.
 * @brief Ensures that the Info notification is displayed properly.
 *
 * @details This test triggers the showNotification method with an Info-level
 * message and checks that the widget remains visible during the fade-out period.
 *
 * @see NotificationOverlay::showNotification
 */
TEST_F(NotificationOverlayTest, ShowInfoNotification) {
	overlay->showNotification("Info message", NotificationLevel::Info, 100);
	QTest::qWait(200);  // Wait for fadeOutTimer to trigger

	// Expect widget to still be visible before fade completes
	EXPECT_TRUE(overlay->isVisible());
}

/*!
 * @test Tests the showNotification method with Warning level.
 * @brief Ensures that the Warning notification fades out and hides the widget.
 *
 * @details This test displays a Warning-level notification and waits long
 * enough for both fade-in and fade-out animations to complete. It then verifies
 * that the widget is hidden.
 *
 * @see NotificationOverlay::showNotification
 */
TEST_F(NotificationOverlayTest, ShowWarningNotificationAndFadeOut) {
	overlay->showNotification("Warning!", NotificationLevel::Warning, 100);
	QTest::qWait(700);  // Wait enough for fade in + fade out

	// After full fade-out, it should be hidden
	EXPECT_FALSE(overlay->isVisible());
}
