#ifndef BATTERYCONTROLLER_HPP
#define BATTERYCONTROLLER_HPP

#include "I2CController.hpp"
#include <QObject>

class BatteryController : public QObject, private I2CController {
  Q_OBJECT

public:
  /**
   * Constructor for the BatteryController class.
   * Initializes the battery controller by setting up I2C communication and
   * calibration.
   *
   * @param i2c_device The path to the I2C device (e.g., "/dev/i2c-1").
   * @param address The I2C address of the battery sensor.
   * @param parent The parent QObject.
   */
  explicit BatteryController(const char *i2c_device, int address,
                             QObject *parent = nullptr);

  /**
   * Destructor for the BatteryController class.
   * Cleans up resources when the controller is no longer needed.
   */
  ~BatteryController() override = default;

  /**
   * Returns the current battery percentage based on bus and shunt voltages.
   * It calculates the battery percentage by measuring the bus voltage and shunt
   * voltage.
   *
   * @return The battery percentage (0 to 100).
   */
  Q_INVOKABLE float getBatteryPercentage();

private:
  /**
   * Configures the INA219 sensor for a 32V, 2A calibration.
   * This function sets the calibration value for the INA219 sensor.
   */
  void setCalibration32V2A();

  /**
   * Reads the bus voltage from the INA219 sensor.
   * The bus voltage is the voltage from the power supply.
   *
   * @return The bus voltage in volts.
   */
  float getBusVoltage_V();

  /**
   * Reads the shunt voltage from the INA219 sensor.
   * The shunt voltage is the voltage drop across the current sense resistor.
   *
   * @return The shunt voltage in volts.
   */
  float getShuntVoltage_V();
};

#endif // BATTERYCONTROLLER_HPP
