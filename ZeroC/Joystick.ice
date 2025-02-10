// Joystick.ice
module Data {
    interface CarData {
        void setJoystickValue(bool newValue);
        bool getJoystickValue();

        void setCarTemperatureValue(double newValue);
        double getCarTemperatureValue();
    };
};

