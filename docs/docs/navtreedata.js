/*
 @licstart  The following is the entire license notice for the JavaScript code in this file.

 The MIT License (MIT)

 Copyright (C) 1997-2020 by Dimitri van Heesch

 Permission is hereby granted, free of charge, to any person obtaining a copy of this software
 and associated documentation files (the "Software"), to deal in the Software without restriction,
 including without limitation the rights to use, copy, modify, merge, publish, distribute,
 sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is
 furnished to do so, subject to the following conditions:

 The above copyright notice and this permission notice shall be included in all copies or
 substantial portions of the Software.

 THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING
 BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND
 NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM,
 DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

 @licend  The above is the entire license notice for the JavaScript code in this file
*/
var NAVTREE =
[
  [ "Hotwheels-Cluster", "index.html", [
    [ "HotWheels 🏁 🏎️", "index.html", "index" ],
    [ "Nvidea or RaspberryPi", "md_ADR_2001-NvideaOrRaspberry.html", [
      [ "Context", "md_ADR_2001-NvideaOrRaspberry.html#autotoc_md1", null ],
      [ "Decision", "md_ADR_2001-NvideaOrRaspberry.html#autotoc_md2", null ],
      [ "Consequences", "md_ADR_2001-NvideaOrRaspberry.html#autotoc_md3", null ]
    ] ],
    [ "Middleware choice", "md_ADR_2MiddlewareDecision.html", [
      [ "ZeroC vs ZeroMQ", "md_ADR_2MiddlewareDecision.html#autotoc_md5", null ]
    ] ],
    [ "TensorRTInferencer Unit Tests", "md_apps_2car__controls_2tests_2documentation_2Inference__TestDoc.html", [
      [ "✅ What’s Covered", "md_apps_2car__controls_2tests_2documentation_2Inference__TestDoc.html#autotoc_md7", null ],
      [ "Test Descriptions", "md_apps_2car__controls_2tests_2documentation_2Inference__TestDoc.html#autotoc_md9", [
        [ "Engine Initialization", "md_apps_2car__controls_2tests_2documentation_2Inference__TestDoc.html#autotoc_md10", null ],
        [ "Image Preprocessing (preprocessImage)", "md_apps_2car__controls_2tests_2documentation_2Inference__TestDoc.html#autotoc_md12", null ],
        [ "Inference Execution (runInference)", "md_apps_2car__controls_2tests_2documentation_2Inference__TestDoc.html#autotoc_md14", null ],
        [ "Output Validation", "md_apps_2car__controls_2tests_2documentation_2Inference__TestDoc.html#autotoc_md16", null ],
        [ "Robustness & Edge Cases", "md_apps_2car__controls_2tests_2documentation_2Inference__TestDoc.html#autotoc_md18", null ]
      ] ],
      [ "Requirements", "md_apps_2car__controls_2tests_2documentation_2Inference__TestDoc.html#autotoc_md20", null ]
    ] ],
    [ "PeripheralController Unit Tests", "md_apps_2car__controls_2tests_2documentation_2PeripheralController__TestDoc.html", [
      [ "✅ What’s Covered", "md_apps_2car__controls_2tests_2documentation_2PeripheralController__TestDoc.html#autotoc_md23", null ],
      [ "Test Descriptions", "md_apps_2car__controls_2tests_2documentation_2PeripheralController__TestDoc.html#autotoc_md25", [
        [ "PWM Functionality", "md_apps_2car__controls_2tests_2documentation_2PeripheralController__TestDoc.html#autotoc_md26", null ],
        [ "Peripheral Initialization", "md_apps_2car__controls_2tests_2documentation_2PeripheralController__TestDoc.html#autotoc_md28", null ],
        [ "I2C Communication", "md_apps_2car__controls_2tests_2documentation_2PeripheralController__TestDoc.html#autotoc_md30", null ],
        [ "Exception Handling", "md_apps_2car__controls_2tests_2documentation_2PeripheralController__TestDoc.html#autotoc_md32", null ]
      ] ],
      [ "🔧 How It Works", "md_apps_2car__controls_2tests_2documentation_2PeripheralController__TestDoc.html#autotoc_md34", null ],
      [ "Requirements", "md_apps_2car__controls_2tests_2documentation_2PeripheralController__TestDoc.html#autotoc_md36", null ]
    ] ],
    [ "Integration Tests: CanBusManager", "md_apps_2cluster_2app__tests_2integration_2documentation_2CanBusManager__TestDoc.html", [
      [ "Overview", "md_apps_2cluster_2app__tests_2integration_2documentation_2CanBusManager__TestDoc.html#autotoc_md38", null ],
      [ "✅ Test Cases", "md_apps_2cluster_2app__tests_2integration_2documentation_2CanBusManager__TestDoc.html#autotoc_md40", [
        [ "ForwardSpeedDataFromMCP2515", "md_apps_2cluster_2app__tests_2integration_2documentation_2CanBusManager__TestDoc.html#autotoc_md41", null ],
        [ "ForwardRpmDataFromMCP2515", "md_apps_2cluster_2app__tests_2integration_2documentation_2CanBusManager__TestDoc.html#autotoc_md42", null ],
        [ "InitializeCanBusManager", "md_apps_2cluster_2app__tests_2integration_2documentation_2CanBusManager__TestDoc.html#autotoc_md43", null ],
        [ "ManagerCleanUpBehavior", "md_apps_2cluster_2app__tests_2integration_2documentation_2CanBusManager__TestDoc.html#autotoc_md44", null ]
      ] ],
      [ "Notes", "md_apps_2cluster_2app__tests_2integration_2documentation_2CanBusManager__TestDoc.html#autotoc_md46", null ]
    ] ],
    [ "Integration Tests: DataManager", "md_apps_2cluster_2app__tests_2integration_2documentation_2DataManager__TestDoc.html", [
      [ "Overview", "md_apps_2cluster_2app__tests_2integration_2documentation_2DataManager__TestDoc.html#autotoc_md48", null ],
      [ "✅ Test Cases", "md_apps_2cluster_2app__tests_2integration_2documentation_2DataManager__TestDoc.html#autotoc_md50", [
        [ "ForwardSpeedDataToVehicleDataManager", "md_apps_2cluster_2app__tests_2integration_2documentation_2DataManager__TestDoc.html#autotoc_md51", null ],
        [ "ForwardRpmDataToVehicleDataManager", "md_apps_2cluster_2app__tests_2integration_2documentation_2DataManager__TestDoc.html#autotoc_md52", null ],
        [ "ForwardSteeringDataToVehicleDataManager", "md_apps_2cluster_2app__tests_2integration_2documentation_2DataManager__TestDoc.html#autotoc_md53", null ],
        [ "ForwardDirectionDataToVehicleDataManager", "md_apps_2cluster_2app__tests_2integration_2documentation_2DataManager__TestDoc.html#autotoc_md54", null ],
        [ "ForwardTimeDataToSystemDataManager", "md_apps_2cluster_2app__tests_2integration_2documentation_2DataManager__TestDoc.html#autotoc_md55", null ],
        [ "ForwardTemperatureDataToSystemDataManager", "md_apps_2cluster_2app__tests_2integration_2documentation_2DataManager__TestDoc.html#autotoc_md56", null ],
        [ "ForwardBatteryPercentageToSystemDataManager", "md_apps_2cluster_2app__tests_2integration_2documentation_2DataManager__TestDoc.html#autotoc_md57", null ],
        [ "ForwardMileageUpdateToVehicleDataManager", "md_apps_2cluster_2app__tests_2integration_2documentation_2DataManager__TestDoc.html#autotoc_md58", null ],
        [ "ToggleDrivingMode", "md_apps_2cluster_2app__tests_2integration_2documentation_2DataManager__TestDoc.html#autotoc_md59", null ],
        [ "ToggleClusterTheme", "md_apps_2cluster_2app__tests_2integration_2documentation_2DataManager__TestDoc.html#autotoc_md60", null ],
        [ "ToggleClusterMetrics", "md_apps_2cluster_2app__tests_2integration_2documentation_2DataManager__TestDoc.html#autotoc_md61", null ]
      ] ],
      [ "Notes", "md_apps_2cluster_2app__tests_2integration_2documentation_2DataManager__TestDoc.html#autotoc_md63", null ]
    ] ],
    [ "Integration Tests: MCP2515Controller", "md_apps_2cluster_2app__tests_2integration_2documentation_2MCP2515Device__TestDoc.html", [
      [ "Overview", "md_apps_2cluster_2app__tests_2integration_2documentation_2MCP2515Device__TestDoc.html#autotoc_md65", null ],
      [ "✅ Test Cases", "md_apps_2cluster_2app__tests_2integration_2documentation_2MCP2515Device__TestDoc.html#autotoc_md67", [
        [ "DataFrameTest", "md_apps_2cluster_2app__tests_2integration_2documentation_2MCP2515Device__TestDoc.html#autotoc_md68", null ],
        [ "RemoteFrameTest", "md_apps_2cluster_2app__tests_2integration_2documentation_2MCP2515Device__TestDoc.html#autotoc_md70", null ],
        [ "ErrorFrameTest", "md_apps_2cluster_2app__tests_2integration_2documentation_2MCP2515Device__TestDoc.html#autotoc_md72", null ],
        [ "MaxBusSpeedTest", "md_apps_2cluster_2app__tests_2integration_2documentation_2MCP2515Device__TestDoc.html#autotoc_md74", null ],
        [ "MinBusSpeedTest", "md_apps_2cluster_2app__tests_2integration_2documentation_2MCP2515Device__TestDoc.html#autotoc_md76", null ]
      ] ],
      [ "Notes", "md_apps_2cluster_2app__tests_2integration_2documentation_2MCP2515Device__TestDoc.html#autotoc_md78", null ]
    ] ],
    [ "Integration Tests: MileageManager", "md_apps_2cluster_2app__tests_2integration_2documentation_2MileageManager__TestDoc.html", [
      [ "Overview", "md_apps_2cluster_2app__tests_2integration_2documentation_2MileageManager__TestDoc.html#autotoc_md80", null ],
      [ "✅ Test Cases", "md_apps_2cluster_2app__tests_2integration_2documentation_2MileageManager__TestDoc.html#autotoc_md82", [
        [ "ForwardMileageData", "md_apps_2cluster_2app__tests_2integration_2documentation_2MileageManager__TestDoc.html#autotoc_md83", null ],
        [ "InitializeMileageManager", "md_apps_2cluster_2app__tests_2integration_2documentation_2MileageManager__TestDoc.html#autotoc_md84", null ],
        [ "UpdateMileageOnSpeedUpdate", "md_apps_2cluster_2app__tests_2integration_2documentation_2MileageManager__TestDoc.html#autotoc_md85", null ],
        [ "SaveMileage", "md_apps_2cluster_2app__tests_2integration_2documentation_2MileageManager__TestDoc.html#autotoc_md86", null ],
        [ "UpdateTimerInterval", "md_apps_2cluster_2app__tests_2integration_2documentation_2MileageManager__TestDoc.html#autotoc_md87", null ],
        [ "ShutdownMileageManager", "md_apps_2cluster_2app__tests_2integration_2documentation_2MileageManager__TestDoc.html#autotoc_md88", null ]
      ] ],
      [ "Notes", "md_apps_2cluster_2app__tests_2integration_2documentation_2MileageManager__TestDoc.html#autotoc_md90", null ]
    ] ],
    [ "Integration Tests: SystemManager", "md_apps_2cluster_2app__tests_2integration_2documentation_2SystemManager__TestDoc.html", [
      [ "Overview", "md_apps_2cluster_2app__tests_2integration_2documentation_2SystemManager__TestDoc.html#autotoc_md92", null ],
      [ "✅ Test Cases", "md_apps_2cluster_2app__tests_2integration_2documentation_2SystemManager__TestDoc.html#autotoc_md94", [
        [ "UpdateTimeSignal", "md_apps_2cluster_2app__tests_2integration_2documentation_2SystemManager__TestDoc.html#autotoc_md95", null ],
        [ "UpdateWifiStatusSignal", "md_apps_2cluster_2app__tests_2integration_2documentation_2SystemManager__TestDoc.html#autotoc_md96", null ],
        [ "UpdateTemperatureSignal", "md_apps_2cluster_2app__tests_2integration_2documentation_2SystemManager__TestDoc.html#autotoc_md97", null ],
        [ "UpdateBatteryPercentageSignal", "md_apps_2cluster_2app__tests_2integration_2documentation_2SystemManager__TestDoc.html#autotoc_md98", null ],
        [ "ShutdownSystemManager", "md_apps_2cluster_2app__tests_2integration_2documentation_2SystemManager__TestDoc.html#autotoc_md99", null ]
      ] ],
      [ "Notes", "md_apps_2cluster_2app__tests_2integration_2documentation_2SystemManager__TestDoc.html#autotoc_md101", null ]
    ] ],
    [ "Unit Tests: CanBusManager", "md_apps_2cluster_2app__tests_2unit_2canbus_2documentation_2CanBusManager__TestsDoc.html", [
      [ "Overview", "md_apps_2cluster_2app__tests_2unit_2canbus_2documentation_2CanBusManager__TestsDoc.html#autotoc_md114", null ],
      [ "✅ Test Cases", "md_apps_2cluster_2app__tests_2unit_2canbus_2documentation_2CanBusManager__TestsDoc.html#autotoc_md116", [
        [ "SpeedSignalEmitsCorrectly", "md_apps_2cluster_2app__tests_2unit_2canbus_2documentation_2CanBusManager__TestsDoc.html#autotoc_md117", null ],
        [ "RpmSignalEmitsCorrectly", "md_apps_2cluster_2app__tests_2unit_2canbus_2documentation_2CanBusManager__TestsDoc.html#autotoc_md119", null ],
        [ "InitializeFailsWhenControllerFails", "md_apps_2cluster_2app__tests_2unit_2canbus_2documentation_2CanBusManager__TestsDoc.html#autotoc_md121", null ],
        [ "InitializeSucceedsWhenControllerSucceeds", "md_apps_2cluster_2app__tests_2unit_2canbus_2documentation_2CanBusManager__TestsDoc.html#autotoc_md123", null ],
        [ "DestructorCallsStopReading", "md_apps_2cluster_2app__tests_2unit_2canbus_2documentation_2CanBusManager__TestsDoc.html#autotoc_md125", null ]
      ] ],
      [ "Notes", "md_apps_2cluster_2app__tests_2unit_2canbus_2documentation_2CanBusManager__TestsDoc.html#autotoc_md127", null ]
    ] ],
    [ "Unit Tests: CANMessageProcessor", "md_apps_2cluster_2app__tests_2unit_2canbus_2documentation_2CANMessageProcessor__TestDoc.html", [
      [ "Overview", "md_apps_2cluster_2app__tests_2unit_2canbus_2documentation_2CANMessageProcessor__TestDoc.html#autotoc_md129", null ],
      [ "✅ Test Cases", "md_apps_2cluster_2app__tests_2unit_2canbus_2documentation_2CANMessageProcessor__TestDoc.html#autotoc_md131", [
        [ "RegisterHandlerSuccess", "md_apps_2cluster_2app__tests_2unit_2canbus_2documentation_2CANMessageProcessor__TestDoc.html#autotoc_md132", null ],
        [ "RegisterHandlerNullThrowsException", "md_apps_2cluster_2app__tests_2unit_2canbus_2documentation_2CANMessageProcessor__TestDoc.html#autotoc_md134", null ],
        [ "ProcessMessageWithRegisteredHandler", "md_apps_2cluster_2app__tests_2unit_2canbus_2documentation_2CANMessageProcessor__TestDoc.html#autotoc_md136", null ],
        [ "ProcessMessageWithUnregisteredHandlerThrowsException", "md_apps_2cluster_2app__tests_2unit_2canbus_2documentation_2CANMessageProcessor__TestDoc.html#autotoc_md138", null ],
        [ "OverwriteHandlerForSameFrameID", "md_apps_2cluster_2app__tests_2unit_2canbus_2documentation_2CANMessageProcessor__TestDoc.html#autotoc_md140", null ]
      ] ],
      [ "Notes", "md_apps_2cluster_2app__tests_2unit_2canbus_2documentation_2CANMessageProcessor__TestDoc.html#autotoc_md142", null ]
    ] ],
    [ "Unit Tests: MCP2515Configurator", "md_apps_2cluster_2app__tests_2unit_2canbus_2documentation_2MCP2515Configurator__TestDoc.html", [
      [ "Overview", "md_apps_2cluster_2app__tests_2unit_2canbus_2documentation_2MCP2515Configurator__TestDoc.html#autotoc_md144", null ],
      [ "✅ Test Cases", "md_apps_2cluster_2app__tests_2unit_2canbus_2documentation_2MCP2515Configurator__TestDoc.html#autotoc_md146", [
        [ "ResetChipSuccess", "md_apps_2cluster_2app__tests_2unit_2canbus_2documentation_2MCP2515Configurator__TestDoc.html#autotoc_md147", null ],
        [ "ResetChipFailure", "md_apps_2cluster_2app__tests_2unit_2canbus_2documentation_2MCP2515Configurator__TestDoc.html#autotoc_md149", null ],
        [ "ConfigureBaudRate", "md_apps_2cluster_2app__tests_2unit_2canbus_2documentation_2MCP2515Configurator__TestDoc.html#autotoc_md151", null ],
        [ "ConfigureTXBuffer", "md_apps_2cluster_2app__tests_2unit_2canbus_2documentation_2MCP2515Configurator__TestDoc.html#autotoc_md153", null ],
        [ "ConfigureRXBuffer", "md_apps_2cluster_2app__tests_2unit_2canbus_2documentation_2MCP2515Configurator__TestDoc.html#autotoc_md155", null ],
        [ "ConfigureFiltersAndMasks", "md_apps_2cluster_2app__tests_2unit_2canbus_2documentation_2MCP2515Configurator__TestDoc.html#autotoc_md157", null ],
        [ "ConfigureInterrupts", "md_apps_2cluster_2app__tests_2unit_2canbus_2documentation_2MCP2515Configurator__TestDoc.html#autotoc_md159", null ],
        [ "SetMode", "md_apps_2cluster_2app__tests_2unit_2canbus_2documentation_2MCP2515Configurator__TestDoc.html#autotoc_md161", null ],
        [ "VerifyModeSuccess", "md_apps_2cluster_2app__tests_2unit_2canbus_2documentation_2MCP2515Configurator__TestDoc.html#autotoc_md163", null ],
        [ "VerifyModeFailure", "md_apps_2cluster_2app__tests_2unit_2canbus_2documentation_2MCP2515Configurator__TestDoc.html#autotoc_md165", null ],
        [ "ReadCANMessageWithData", "md_apps_2cluster_2app__tests_2unit_2canbus_2documentation_2MCP2515Configurator__TestDoc.html#autotoc_md167", null ],
        [ "ReadCANMessageNoData", "md_apps_2cluster_2app__tests_2unit_2canbus_2documentation_2MCP2515Configurator__TestDoc.html#autotoc_md169", null ]
      ] ],
      [ "Notes", "md_apps_2cluster_2app__tests_2unit_2canbus_2documentation_2MCP2515Configurator__TestDoc.html#autotoc_md171", null ]
    ] ],
    [ "Unit Tests: MCP2515Controller", "md_apps_2cluster_2app__tests_2unit_2canbus_2documentation_2MCP2515Controller__TestDoc.html", [
      [ "Overview", "md_apps_2cluster_2app__tests_2unit_2canbus_2documentation_2MCP2515Controller__TestDoc.html#autotoc_md173", null ],
      [ "✅ Test Cases", "md_apps_2cluster_2app__tests_2unit_2canbus_2documentation_2MCP2515Controller__TestDoc.html#autotoc_md175", [
        [ "InitializationSuccess", "md_apps_2cluster_2app__tests_2unit_2canbus_2documentation_2MCP2515Controller__TestDoc.html#autotoc_md176", null ],
        [ "InitializationFailure", "md_apps_2cluster_2app__tests_2unit_2canbus_2documentation_2MCP2515Controller__TestDoc.html#autotoc_md178", null ],
        [ "SetupHandlersTest", "md_apps_2cluster_2app__tests_2unit_2canbus_2documentation_2MCP2515Controller__TestDoc.html#autotoc_md180", null ],
        [ "SpeedUpdatedSignal", "md_apps_2cluster_2app__tests_2unit_2canbus_2documentation_2MCP2515Controller__TestDoc.html#autotoc_md182", null ],
        [ "RpmUpdatedSignal", "md_apps_2cluster_2app__tests_2unit_2canbus_2documentation_2MCP2515Controller__TestDoc.html#autotoc_md184", null ],
        [ "ProcessReadingCallsHandlers", "md_apps_2cluster_2app__tests_2unit_2canbus_2documentation_2MCP2515Controller__TestDoc.html#autotoc_md186", null ],
        [ "StopReadingStopsProcessing", "md_apps_2cluster_2app__tests_2unit_2canbus_2documentation_2MCP2515Controller__TestDoc.html#autotoc_md188", null ]
      ] ],
      [ "Notes", "md_apps_2cluster_2app__tests_2unit_2canbus_2documentation_2MCP2515Controller__TestDoc.html#autotoc_md190", null ]
    ] ],
    [ "Unit Tests: SPIController", "md_apps_2cluster_2app__tests_2unit_2canbus_2documentation_2SPIController__TestDoc.html", [
      [ "Overview", "md_apps_2cluster_2app__tests_2unit_2canbus_2documentation_2SPIController__TestDoc.html#autotoc_md192", null ],
      [ "✅ Test Cases", "md_apps_2cluster_2app__tests_2unit_2canbus_2documentation_2SPIController__TestDoc.html#autotoc_md194", [
        [ "OpenDeviceSuccess", "md_apps_2cluster_2app__tests_2unit_2canbus_2documentation_2SPIController__TestDoc.html#autotoc_md195", null ],
        [ "OpenDeviceFailure", "md_apps_2cluster_2app__tests_2unit_2canbus_2documentation_2SPIController__TestDoc.html#autotoc_md197", null ],
        [ "ConfigureSPIValidParameters", "md_apps_2cluster_2app__tests_2unit_2canbus_2documentation_2SPIController__TestDoc.html#autotoc_md199", null ],
        [ "WriteByteSuccess", "md_apps_2cluster_2app__tests_2unit_2canbus_2documentation_2SPIController__TestDoc.html#autotoc_md201", null ],
        [ "ReadByteSuccess", "md_apps_2cluster_2app__tests_2unit_2canbus_2documentation_2SPIController__TestDoc.html#autotoc_md203", null ],
        [ "SpiTransferSuccess", "md_apps_2cluster_2app__tests_2unit_2canbus_2documentation_2SPIController__TestDoc.html#autotoc_md205", null ],
        [ "CloseDeviceSuccess", "md_apps_2cluster_2app__tests_2unit_2canbus_2documentation_2SPIController__TestDoc.html#autotoc_md207", null ]
      ] ],
      [ "Notes", "md_apps_2cluster_2app__tests_2unit_2canbus_2documentation_2SPIController__TestDoc.html#autotoc_md209", null ]
    ] ],
    [ "Unit Tests: ClusterSettingsManager", "md_apps_2cluster_2app__tests_2unit_2data_2documentation_2ClusterSettiingsManager__TestDoc.html", [
      [ "Overview", "md_apps_2cluster_2app__tests_2unit_2data_2documentation_2ClusterSettiingsManager__TestDoc.html#autotoc_md211", null ],
      [ "✅ Test Cases", "md_apps_2cluster_2app__tests_2unit_2data_2documentation_2ClusterSettiingsManager__TestDoc.html#autotoc_md213", [
        [ "ToggleDrivingModeEmitsSignal", "md_apps_2cluster_2app__tests_2unit_2data_2documentation_2ClusterSettiingsManager__TestDoc.html#autotoc_md214", null ],
        [ "ToggleClusterThemeEmitsSignal", "md_apps_2cluster_2app__tests_2unit_2data_2documentation_2ClusterSettiingsManager__TestDoc.html#autotoc_md216", null ],
        [ "ToggleClusterMetricsEmitsSignal", "md_apps_2cluster_2app__tests_2unit_2data_2documentation_2ClusterSettiingsManager__TestDoc.html#autotoc_md218", null ]
      ] ],
      [ "Notes", "md_apps_2cluster_2app__tests_2unit_2data_2documentation_2ClusterSettiingsManager__TestDoc.html#autotoc_md220", null ]
    ] ],
    [ "Unit Tests: SystemDataManager", "md_apps_2cluster_2app__tests_2unit_2data_2documentation_2SystemDataManager__TestDoc.html", [
      [ "Overview", "md_apps_2cluster_2app__tests_2unit_2data_2documentation_2SystemDataManager__TestDoc.html#autotoc_md222", null ],
      [ "✅ Test Cases", "md_apps_2cluster_2app__tests_2unit_2data_2documentation_2SystemDataManager__TestDoc.html#autotoc_md224", [
        [ "TimeDataEmitsSignal", "md_apps_2cluster_2app__tests_2unit_2data_2documentation_2SystemDataManager__TestDoc.html#autotoc_md225", null ],
        [ "TemperatureDataEmitsSignalOnChange", "md_apps_2cluster_2app__tests_2unit_2data_2documentation_2SystemDataManager__TestDoc.html#autotoc_md227", null ],
        [ "BatteryPercentageEmitsSignalOnChange", "md_apps_2cluster_2app__tests_2unit_2data_2documentation_2SystemDataManager__TestDoc.html#autotoc_md229", null ]
      ] ],
      [ "Notes", "md_apps_2cluster_2app__tests_2unit_2data_2documentation_2SystemDataManager__TestDoc.html#autotoc_md231", null ]
    ] ],
    [ "Unit Tests: VehicleDataManager", "md_apps_2cluster_2app__tests_2unit_2data_2documentation_2VehicleDataManager__TestDoc.html", [
      [ "Overview", "md_apps_2cluster_2app__tests_2unit_2data_2documentation_2VehicleDataManager__TestDoc.html#autotoc_md233", null ],
      [ "✅ Test Cases", "md_apps_2cluster_2app__tests_2unit_2data_2documentation_2VehicleDataManager__TestDoc.html#autotoc_md235", [
        [ "RpmDataEmitsSignal", "md_apps_2cluster_2app__tests_2unit_2data_2documentation_2VehicleDataManager__TestDoc.html#autotoc_md236", null ],
        [ "SpeedDataEmitsSignalInKilometers", "md_apps_2cluster_2app__tests_2unit_2data_2documentation_2VehicleDataManager__TestDoc.html#autotoc_md238", null ],
        [ "MileageDataEmitsSignalOnChange", "md_apps_2cluster_2app__tests_2unit_2data_2documentation_2VehicleDataManager__TestDoc.html#autotoc_md240", null ],
        [ "DirectionDataEmitsSignalOnChange", "md_apps_2cluster_2app__tests_2unit_2data_2documentation_2VehicleDataManager__TestDoc.html#autotoc_md242", null ],
        [ "SteeringDataEmitsSignalOnChange", "md_apps_2cluster_2app__tests_2unit_2data_2documentation_2VehicleDataManager__TestDoc.html#autotoc_md244", null ]
      ] ],
      [ "Notes", "md_apps_2cluster_2app__tests_2unit_2data_2documentation_2VehicleDataManager__TestDoc.html#autotoc_md246", null ]
    ] ],
    [ "Unit Tests: MileageCalculator", "md_apps_2cluster_2app__tests_2unit_2mileage_2documentation_2MileageCalculator__TestDoc.html", [
      [ "Overview", "md_apps_2cluster_2app__tests_2unit_2mileage_2documentation_2MileageCalculator__TestDoc.html#autotoc_md248", null ],
      [ "✅ Test Cases", "md_apps_2cluster_2app__tests_2unit_2mileage_2documentation_2MileageCalculator__TestDoc.html#autotoc_md250", [
        [ "AddSpeed_DoesNotCrash", "md_apps_2cluster_2app__tests_2unit_2mileage_2documentation_2MileageCalculator__TestDoc.html#autotoc_md251", null ],
        [ "CalculateDistance_NoSpeeds_ReturnsZero", "md_apps_2cluster_2app__tests_2unit_2mileage_2documentation_2MileageCalculator__TestDoc.html#autotoc_md253", null ],
        [ "CalculateDistance_ZeroSpeed_ReturnsZero", "md_apps_2cluster_2app__tests_2unit_2mileage_2documentation_2MileageCalculator__TestDoc.html#autotoc_md255", null ],
        [ "CalculateDistance_BasicCalculation", "md_apps_2cluster_2app__tests_2unit_2mileage_2documentation_2MileageCalculator__TestDoc.html#autotoc_md257", null ],
        [ "CalculateDistance_MultipleSpeeds", "md_apps_2cluster_2app__tests_2unit_2mileage_2documentation_2MileageCalculator__TestDoc.html#autotoc_md259", null ]
      ] ],
      [ "Notes", "md_apps_2cluster_2app__tests_2unit_2mileage_2documentation_2MileageCalculator__TestDoc.html#autotoc_md261", null ]
    ] ],
    [ "Unit Tests: MileageFileHandler", "md_apps_2cluster_2app__tests_2unit_2mileage_2documentation_2MileageFileHandler__TestDoc.html", [
      [ "Overview", "md_apps_2cluster_2app__tests_2unit_2mileage_2documentation_2MileageFileHandler__TestDoc.html#autotoc_md263", null ],
      [ "✅ Test Cases", "md_apps_2cluster_2app__tests_2unit_2mileage_2documentation_2MileageFileHandler__TestDoc.html#autotoc_md265", [
        [ "EnsureFileExists_FileDoesNotExist_CreatesFileWithZeroMileage", "md_apps_2cluster_2app__tests_2unit_2mileage_2documentation_2MileageFileHandler__TestDoc.html#autotoc_md266", null ],
        [ "ReadMileage_ValidNumber_ReturnsParsedValue", "md_apps_2cluster_2app__tests_2unit_2mileage_2documentation_2MileageFileHandler__TestDoc.html#autotoc_md268", null ],
        [ "ReadMileage_InvalidNumber_ReturnsZero", "md_apps_2cluster_2app__tests_2unit_2mileage_2documentation_2MileageFileHandler__TestDoc.html#autotoc_md270", null ],
        [ "WriteMileage_ValidNumber_WritesToFile", "md_apps_2cluster_2app__tests_2unit_2mileage_2documentation_2MileageFileHandler__TestDoc.html#autotoc_md272", null ]
      ] ],
      [ "Notes", "md_apps_2cluster_2app__tests_2unit_2mileage_2documentation_2MileageFileHandler__TestDoc.html#autotoc_md274", null ]
    ] ],
    [ "Unit Tests: MileageManager", "md_apps_2cluster_2app__tests_2unit_2mileage_2documentation_2MileageManager__TestDoc.html", [
      [ "Overview", "md_apps_2cluster_2app__tests_2unit_2mileage_2documentation_2MileageManager__TestDoc.html#autotoc_md276", null ],
      [ "✅ Test Cases", "md_apps_2cluster_2app__tests_2unit_2mileage_2documentation_2MileageManager__TestDoc.html#autotoc_md278", [
        [ "Initialize_LoadsMileageFromFile", "md_apps_2cluster_2app__tests_2unit_2mileage_2documentation_2MileageManager__TestDoc.html#autotoc_md279", null ],
        [ "OnSpeedUpdated_CallsCalculator", "md_apps_2cluster_2app__tests_2unit_2mileage_2documentation_2MileageManager__TestDoc.html#autotoc_md281", null ],
        [ "UpdateMileage_EmitsMileageUpdatedSignal", "md_apps_2cluster_2app__tests_2unit_2mileage_2documentation_2MileageManager__TestDoc.html#autotoc_md283", null ],
        [ "SaveMileage_CallsFileHandler", "md_apps_2cluster_2app__tests_2unit_2mileage_2documentation_2MileageManager__TestDoc.html#autotoc_md285", null ]
      ] ],
      [ "Notes", "md_apps_2cluster_2app__tests_2unit_2mileage_2documentation_2MileageManager__TestDoc.html#autotoc_md287", null ]
    ] ],
    [ "Unit Tests: BatteryController", "md_apps_2cluster_2app__tests_2unit_2system_2documentation_2BatteryController__TestDoc.html", [
      [ "Overview", "md_apps_2cluster_2app__tests_2unit_2system_2documentation_2BatteryController__TestDoc.html#autotoc_md289", null ],
      [ "✅ Test Cases", "md_apps_2cluster_2app__tests_2unit_2system_2documentation_2BatteryController__TestDoc.html#autotoc_md291", [
        [ "Initialization_CallsCalibration", "md_apps_2cluster_2app__tests_2unit_2system_2documentation_2BatteryController__TestDoc.html#autotoc_md292", null ],
        [ "GetBatteryPercentage_CorrectCalculation", "md_apps_2cluster_2app__tests_2unit_2system_2documentation_2BatteryController__TestDoc.html#autotoc_md294", null ]
      ] ],
      [ "Notes", "md_apps_2cluster_2app__tests_2unit_2system_2documentation_2BatteryController__TestDoc.html#autotoc_md296", null ]
    ] ],
    [ "Unit Tests: SystemInfoProvider", "md_apps_2cluster_2app__tests_2unit_2system_2documentation_2SystemInfoProvider__TestDoc.html", [
      [ "Overview", "md_apps_2cluster_2app__tests_2unit_2system_2documentation_2SystemInfoProvider__TestDoc.html#autotoc_md298", null ],
      [ "✅ Test Cases", "md_apps_2cluster_2app__tests_2unit_2system_2documentation_2SystemInfoProvider__TestDoc.html#autotoc_md300", [
        [ "GetWifiStatus_Connected", "md_apps_2cluster_2app__tests_2unit_2system_2documentation_2SystemInfoProvider__TestDoc.html#autotoc_md301", null ],
        [ "GetWifiStatus_Disconnected", "md_apps_2cluster_2app__tests_2unit_2system_2documentation_2SystemInfoProvider__TestDoc.html#autotoc_md303", null ],
        [ "GetWifiStatus_NoInterface", "md_apps_2cluster_2app__tests_2unit_2system_2documentation_2SystemInfoProvider__TestDoc.html#autotoc_md305", null ],
        [ "GetTemperature_ValidReading", "md_apps_2cluster_2app__tests_2unit_2system_2documentation_2SystemInfoProvider__TestDoc.html#autotoc_md307", null ],
        [ "GetTemperature_InvalidReading", "md_apps_2cluster_2app__tests_2unit_2system_2documentation_2SystemInfoProvider__TestDoc.html#autotoc_md309", null ],
        [ "GetIpAddress_Valid", "md_apps_2cluster_2app__tests_2unit_2system_2documentation_2SystemInfoProvider__TestDoc.html#autotoc_md311", null ],
        [ "GetIpAddress_NoIP", "md_apps_2cluster_2app__tests_2unit_2system_2documentation_2SystemInfoProvider__TestDoc.html#autotoc_md313", null ]
      ] ],
      [ "Notes", "md_apps_2cluster_2app__tests_2unit_2system_2documentation_2SystemInfoProvider__TestDoc.html#autotoc_md315", null ]
    ] ],
    [ "Unit Tests: SystemManager", "md_apps_2cluster_2app__tests_2unit_2system_2documentation_2SystemManager__TestDoc.html", [
      [ "Overview", "md_apps_2cluster_2app__tests_2unit_2system_2documentation_2SystemManager__TestDoc.html#autotoc_md317", null ],
      [ "✅ Test Cases", "md_apps_2cluster_2app__tests_2unit_2system_2documentation_2SystemManager__TestDoc.html#autotoc_md319", [
        [ "UpdateTime_EmitsCorrectSignal", "md_apps_2cluster_2app__tests_2unit_2system_2documentation_2SystemManager__TestDoc.html#autotoc_md320", null ],
        [ "UpdateSystemStatus_EmitsWifiStatus", "md_apps_2cluster_2app__tests_2unit_2system_2documentation_2SystemManager__TestDoc.html#autotoc_md322", null ],
        [ "UpdateSystemStatus_EmitsTemperature", "md_apps_2cluster_2app__tests_2unit_2system_2documentation_2SystemManager__TestDoc.html#autotoc_md324", null ],
        [ "UpdateSystemStatus_EmitsBatteryPercentage", "md_apps_2cluster_2app__tests_2unit_2system_2documentation_2SystemManager__TestDoc.html#autotoc_md326", null ]
      ] ],
      [ "Notes", "md_apps_2cluster_2app__tests_2unit_2system_2documentation_2SystemManager__TestDoc.html#autotoc_md328", null ]
    ] ],
    [ "HotWheels - Instrument Cluster ⏲ 🎮", "md_Instrument-Cluster.html", [
      [ "Module Description", "md_Instrument-Cluster.html#autotoc_md341", null ],
      [ "Communication Architecture", "md_Instrument-Cluster.html#autotoc_md342", null ],
      [ "Software Architecture", "md_Instrument-Cluster.html#autotoc_md343", null ],
      [ "Adaptation to Read CAN Messages Using SPI Pins", "md_Instrument-Cluster.html#autotoc_md344", null ],
      [ "Cross-Compilation Method", "md_Instrument-Cluster.html#autotoc_md345", null ],
      [ "Results", "md_Instrument-Cluster.html#autotoc_md346", null ]
    ] ],
    [ "Test List", "test.html", null ],
    [ "Namespaces", "namespaces.html", [
      [ "Namespace List", "namespaces.html", "namespaces_dup" ],
      [ "Namespace Members", "namespacemembers.html", [
        [ "All", "namespacemembers.html", null ],
        [ "Functions", "namespacemembers_func.html", null ],
        [ "Variables", "namespacemembers_vars.html", null ]
      ] ]
    ] ],
    [ "Data Structures", "annotated.html", [
      [ "Data Structures", "annotated.html", "annotated_dup" ],
      [ "Data Structure Index", "classes.html", null ],
      [ "Class Hierarchy", "hierarchy.html", "hierarchy" ],
      [ "Data Fields", "functions.html", [
        [ "All", "functions.html", "functions_dup" ],
        [ "Functions", "functions_func.html", "functions_func" ],
        [ "Variables", "functions_vars.html", "functions_vars" ],
        [ "Typedefs", "functions_type.html", null ],
        [ "Enumerations", "functions_enum.html", null ]
      ] ]
    ] ],
    [ "Files", "files.html", [
      [ "File List", "files.html", "files_dup" ],
      [ "Globals", "globals.html", [
        [ "All", "globals.html", null ],
        [ "Functions", "globals_func.html", null ],
        [ "Variables", "globals_vars.html", null ],
        [ "Typedefs", "globals_type.html", null ],
        [ "Enumerations", "globals_enum.html", null ],
        [ "Macros", "globals_defs.html", null ]
      ] ]
    ] ]
  ] ]
];

var NAVTREEINDEX =
[
"BatteryController_8cpp.html",
"classBatteryController.html#a2af3e0526f5bf22a6d13d79567afac6b",
"classIMileageFileHandler.html#ad2ffebd67fa213a902c8fc999ebf1589",
"classPeripheralController.html#a74944b38fa602d38df1d7d3fd5c3e8bf",
"cluster_2includes_2data_2enums_8hpp.html#ad415a8d90ae9e75a02f0ff53c32c1cdaaa18366b217ebf811ad1886e4f4f865b2",
"md_apps_2cluster_2app__tests_2unit_2canbus_2documentation_2CanBusManager__TestsDoc.html#autotoc_md117",
"test__PeripheralController_8cpp.html#a8b8adbb98eaf80db80ceff5ae32577fe"
];

var SYNCONMSG = 'click to disable panel synchronization';
var SYNCOFFMSG = 'click to enable panel synchronization';