import smbus2
import time
import math

class JetCar:
    def __init__(self, servo_addr=0x40, motor_addr=0x60):
        # Servo setup
        self.servo_bus = smbus2.SMBus(1)
        self.SERVO_ADDR = servo_addr
        self.STEERING_CHANNEL = 0

        # Motor controller setup
        self.motor_bus = smbus2.SMBus(1)
        self.MOTOR_ADDR = motor_addr

        #find /usr/include -name i2c-dev.h
        # Initialize both systems
        self.init_servo()
        self.init_motors()

    def init_servo(self):
        try:
            # Reset PCA9685
            self.servo_bus.write_byte_data(self.SERVO_ADDR, 0x00, 0x06)
            time.sleep(0.1)

            # Setup servo control
            self.servo_bus.write_byte_data(self.SERVO_ADDR, 0x00, 0x10)
            time.sleep(0.1)

            # Set frequency (~50Hz)
            self.servo_bus.write_byte_data(self.SERVO_ADDR, 0xFE, 0x79)
            time.sleep(0.1)

            # Configure MODE2
            self.servo_bus.write_byte_data(self.SERVO_ADDR, 0x01, 0x04)
            time.sleep(0.1)

            # Enable auto-increment
            self.servo_bus.write_byte_data(self.SERVO_ADDR, 0x00, 0x20)
            time.sleep(0.1)

            return True
        except Exception as e:
            print(f"Servo init error: {e}")
            return False

    def init_motors(self):
        try:
            # Configure motor controller
            self.motor_bus.write_byte_data(self.MOTOR_ADDR, 0x00, 0x20)

            # Set frequency to 60Hz
            prescale = int(math.floor(25000000.0 / 4096.0 / 100 - 1))
            oldmode = self.motor_bus.read_byte_data(self.MOTOR_ADDR, 0x00)
            newmode = (oldmode & 0x7F) | 0x10
            self.motor_bus.write_byte_data(self.MOTOR_ADDR, 0x00, newmode)
            self.motor_bus.write_byte_data(self.MOTOR_ADDR, 0xFE, prescale)
            self.motor_bus.write_byte_data(self.MOTOR_ADDR, 0x00, oldmode)
            time.sleep(0.005)
            self.motor_bus.write_byte_data(self.MOTOR_ADDR, 0x00, oldmode | 0xa1)

            return True
        except Exception as e:
            print(f"Motor init error: {e}")
            return False

try:
    car = JetCar()
except Exception as e:
    print(f"Error initializing JetCar: {e}")
