#include <gtest/gtest.h>
#include <gmock/gmock.h>
#include <QApplication>
#include <QLabel>
#include <QPushButton>
#include <QToolButton>
#include "DisplayManager.hpp"

class FakeUI : public Ui::CarManager {
public:
	FakeUI() {
		speedLabel = new QLabel;
		timeLabel = new QLabel;
		dateLabel = new QLabel;
		temperatureLabel = new QLabel;
		batteryLabel = new QLabel;
		mileageLabel = new QLabel;
		speedMetricsLabel = new QLabel;
		drivingModeLabel = new QLabel;

		directionDriveLabel = new QLabel;
		directionNeutralLabel = new QLabel;
		directionReverseLabel = new QLabel;
		directionParkLabel = new QLabel;

		laneKeepingAssistLabel = new QLabel;

		speedLimit50Label = new QLabel;
		speedLimit80Label = new QLabel;

		wifiToggleButton = new QToolButton;
		toggleDrivingModeButton = new QPushButton;
		toggleMetricsButton = new QPushButton;

		inferenceLabel = new QLabel;
	}

	~FakeUI() {
		delete speedLabel;
		delete timeLabel;
		delete dateLabel;
		delete temperatureLabel;
		delete batteryLabel;
		delete mileageLabel;
		delete speedMetricsLabel;
		delete drivingModeLabel;
		delete directionDriveLabel;
		delete directionNeutralLabel;
		delete directionReverseLabel;
		delete directionParkLabel;
		delete laneKeepingAssistLabel;
		delete speedLimit50Label;
		delete speedLimit80Label;
		delete wifiToggleButton;
		delete toggleDrivingModeButton;
		delete toggleMetricsButton;
		delete inferenceLabel;
	}
};

class DisplayManagerTest : public ::testing::Test {
protected:
	FakeUI* fakeUI;
	DisplayManager* displayManager;

	void SetUp() override {
		fakeUI = new FakeUI;
		displayManager = new DisplayManager(fakeUI);
	}

	void TearDown() override {
		delete displayManager;
		delete fakeUI;
	}
};

/*!
 * @test Tests updateCanBusData sets speed label correctly.
 * @brief Ensures speed label is updated from CAN bus data.
 *
 * @details Verifies the label text is converted and updated properly.
 *
 * @see DisplayManager::updateCanBusData
 */
TEST_F(DisplayManagerTest, UpdateCanBusData_SetsSpeedLabel) {
	displayManager->updateCanBusData(123.4f, 0);
	EXPECT_EQ(fakeUI->speedLabel->text(), "123");
}

/*!
 * @test Tests updateSystemTime updates time and date labels.
 * @brief Ensures system time is set correctly on UI.
 *
 * @see DisplayManager::updateSystemTime
 */
TEST_F(DisplayManagerTest, UpdateSystemTime_UpdatesLabels) {
	displayManager->updateSystemTime("June", "14:52", "Wednesday");
	EXPECT_EQ(fakeUI->dateLabel->text(), "June Wednesday");
	EXPECT_EQ(fakeUI->timeLabel->text(), "14:52");
}

/*!
 * @test Tests updateBatteryPercentage displays correct percentage.
 * @brief Ensures label shows percentage correctly formatted.
 *
 * @see DisplayManager::updateBatteryPercentage
 */
TEST_F(DisplayManagerTest, UpdateBatteryPercentage_SetsText) {
	displayManager->updateBatteryPercentage(67.3f);
	EXPECT_EQ(fakeUI->batteryLabel->text(), "67%");
}

/*!
 * @test Tests updateMileage sets mileage label properly.
 * @brief Ensures mileage is shown with "m" suffix.
 *
 * @see DisplayManager::updateMileage
 */
TEST_F(DisplayManagerTest, UpdateMileage_SetsMileageLabel) {
	displayManager->updateMileage(312.7);
	EXPECT_EQ(fakeUI->mileageLabel->text(), "312 m");
}

/*!
 * @test Tests updateTemperature sets temperature label.
 * @brief Verifies temperature string is applied directly.
 *
 * @see DisplayManager::updateTemperature
 */
TEST_F(DisplayManagerTest, UpdateTemperature_SetsLabelText) {
	displayManager->updateTemperature("23°C");
	EXPECT_EQ(fakeUI->temperatureLabel->text(), "23°C");
}

/*!
 * @test Tests updateClusterMetrics switches between KM/H and MPH.
 * @brief Confirms metrics switch correctly.
 *
 * @see DisplayManager::updateClusterMetrics
 */
TEST_F(DisplayManagerTest, UpdateClusterMetrics_SetsText) {
	displayManager->updateClusterMetrics(ClusterMetrics::Kilometers);
	EXPECT_EQ(fakeUI->speedMetricsLabel->text(), "KM/H");

	displayManager->updateClusterMetrics(ClusterMetrics::Miles);
	EXPECT_EQ(fakeUI->speedMetricsLabel->text(), "MPH");
}

//NEW ONES

TEST_F(DisplayManagerTest, UpdateEngineData_UpdatesDriveStyle) {
	displayManager->updateEngineData(CarDirection::Drive, 0);
	EXPECT_EQ(fakeUI->directionDriveLabel->styleSheet(), "color: green; background: transparent;");
	EXPECT_EQ(fakeUI->directionNeutralLabel->styleSheet(), "color: white; background: transparent;");
}

TEST_F(DisplayManagerTest, UpdateEngineData_UpdatesReverseStyle) {
	displayManager->updateEngineData(CarDirection::Reverse, 0);
	EXPECT_EQ(fakeUI->directionReverseLabel->styleSheet(), "color: lightgreen; background: transparent;");
	EXPECT_EQ(fakeUI->directionDriveLabel->styleSheet(), "color: white; background: transparent;");
}

TEST_F(DisplayManagerTest, UpdateDrivingMode_SwitchToManual) {
	displayManager->updateDrivingMode(DrivingMode::Manual);
	EXPECT_EQ(fakeUI->drivingModeLabel->text(), "Manual");
	EXPECT_TRUE(fakeUI->laneKeepingAssistLabel->isVisible());
}

TEST_F(DisplayManagerTest, UpdateDrivingMode_SwitchToAutomatic) {
	displayManager->updateDrivingMode(DrivingMode::Automatic);
	EXPECT_EQ(fakeUI->drivingModeLabel->text(), "Automatic");
	EXPECT_FALSE(fakeUI->laneKeepingAssistLabel->isVisible());
	// Can't test blinking visibility toggle in unit test without waiting/timers
}

TEST_F(DisplayManagerTest, UpdateClusterTheme_NoCrash) {
	EXPECT_NO_THROW(displayManager->updateClusterTheme(ClusterTheme::Dark));
	EXPECT_NO_THROW(displayManager->updateClusterTheme(ClusterTheme::Light));
}

TEST_F(DisplayManagerTest, UpdateSpeedLimitLabels_Shows50) {
	displayManager->updateSpeedLimitLabels(50);
	EXPECT_TRUE(fakeUI->speedLimit50Label->isVisible());
	EXPECT_FALSE(fakeUI->speedLimit80Label->isVisible());
}

TEST_F(DisplayManagerTest, UpdateSpeedLimitLabels_Shows80) {
	displayManager->updateSpeedLimitLabels(80);
	EXPECT_TRUE(fakeUI->speedLimit80Label->isVisible());
	EXPECT_FALSE(fakeUI->speedLimit50Label->isVisible());
}

TEST_F(DisplayManagerTest, DisplayInferenceImage_SetsPixmap) {
	QImage image(100, 100, QImage::Format_ARGB32);
	image.fill(Qt::red);
	displayManager->displayInferenceImage(image);
	EXPECT_FALSE(fakeUI->inferenceLabel->pixmap(Qt::ReturnByValue).isNull());
}

TEST_F(DisplayManagerTest, GetWifiSSID_ReturnsString) {
	QString ssid = displayManager->getWifiSSID();
	EXPECT_FALSE(ssid.isEmpty());  // Should be "Not connected" or a valid SSID
}

TEST_F(DisplayManagerTest, GetLocalIPAddress_ReturnsString) {
	QString ip = displayManager->getLocalIPAddress();
	EXPECT_FALSE(ip.isEmpty());

	if (ip != "No IP") {
		// Optionally check format
		EXPECT_TRUE(ip.contains('.'));  // Basic IPv4 check
	}
}
