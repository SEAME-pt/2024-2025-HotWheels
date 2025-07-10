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
    [ "Application Test Suite", "index.html", [
      [ "📁 Directory Structure", "index.html#autotoc_md104", null ],
      [ "Testing Categories", "index.html#autotoc_md106", [
        [ "1. Unit Tests (<tt>unit/</tt>)", "index.html#autotoc_md107", null ],
        [ "2. Integration Tests (<tt>integration/</tt>)", "index.html#autotoc_md108", null ],
        [ "3. Functional Tests (<tt>functional/</tt>)", "index.html#autotoc_md109", null ],
        [ "4. Mock Implementations (<tt>mocks/</tt>)", "index.html#autotoc_md110", null ]
      ] ],
      [ "Test Documentation", "index.html#autotoc_md112", null ]
    ] ],
    [ "Nvidea or RaspberryPi", "md_ADR_001_NvideaOrRaspberry.html", [
      [ "Context", "md_ADR_001_NvideaOrRaspberry.html#autotoc_md1", null ],
      [ "Decision", "md_ADR_001_NvideaOrRaspberry.html#autotoc_md2", null ],
      [ "Consequences", "md_ADR_001_NvideaOrRaspberry.html#autotoc_md3", null ]
    ] ],
    [ "Middleware choice", "md_ADR_MiddlewareDecision.html", [
      [ "ZeroC vs ZeroMQ", "md_ADR_MiddlewareDecision.html#autotoc_md5", null ]
    ] ],
    [ "TensorRTInferencer Unit Tests", "md_apps_car_controls_tests_documentation_Inference_TestDoc.html", [
      [ "✅ What’s Covered", "md_apps_car_controls_tests_documentation_Inference_TestDoc.html#autotoc_md7", null ],
      [ "Test Descriptions", "md_apps_car_controls_tests_documentation_Inference_TestDoc.html#autotoc_md9", [
        [ "Engine Initialization", "md_apps_car_controls_tests_documentation_Inference_TestDoc.html#autotoc_md10", null ],
        [ "Image Preprocessing (<tt>preprocessImage</tt>)", "md_apps_car_controls_tests_documentation_Inference_TestDoc.html#autotoc_md12", null ],
        [ "Inference Execution (<tt>runInference</tt>)", "md_apps_car_controls_tests_documentation_Inference_TestDoc.html#autotoc_md14", null ],
        [ "Output Validation", "md_apps_car_controls_tests_documentation_Inference_TestDoc.html#autotoc_md16", null ],
        [ "Robustness & Edge Cases", "md_apps_car_controls_tests_documentation_Inference_TestDoc.html#autotoc_md18", null ]
      ] ],
      [ "Requirements", "md_apps_car_controls_tests_documentation_Inference_TestDoc.html#autotoc_md20", null ]
    ] ],
    [ "PeripheralController Unit Tests", "md_apps_car_controls_tests_documentation_PeripheralController_TestDoc.html", [
      [ "✅ What’s Covered", "md_apps_car_controls_tests_documentation_PeripheralController_TestDoc.html#autotoc_md23", null ],
      [ "Test Descriptions", "md_apps_car_controls_tests_documentation_PeripheralController_TestDoc.html#autotoc_md25", [
        [ "PWM Functionality", "md_apps_car_controls_tests_documentation_PeripheralController_TestDoc.html#autotoc_md26", null ],
        [ "Peripheral Initialization", "md_apps_car_controls_tests_documentation_PeripheralController_TestDoc.html#autotoc_md28", null ],
        [ "I2C Communication", "md_apps_car_controls_tests_documentation_PeripheralController_TestDoc.html#autotoc_md30", null ],
        [ "Exception Handling", "md_apps_car_controls_tests_documentation_PeripheralController_TestDoc.html#autotoc_md32", null ]
      ] ],
      [ "🔧 How It Works", "md_apps_car_controls_tests_documentation_PeripheralController_TestDoc.html#autotoc_md34", null ],
      [ "Requirements", "md_apps_car_controls_tests_documentation_PeripheralController_TestDoc.html#autotoc_md36", null ]
    ] ],
    [ "Integration Tests: CanBusManager", "md_apps_cluster_app_tests_integration_documentation_CanBusManager_TestDoc.html", [
      [ "Overview", "md_apps_cluster_app_tests_integration_documentation_CanBusManager_TestDoc.html#autotoc_md38", null ],
      [ "✅ Test Cases", "md_apps_cluster_app_tests_integration_documentation_CanBusManager_TestDoc.html#autotoc_md40", [
        [ "1. <tt>ForwardSpeedDataFromMCP2515</tt>", "md_apps_cluster_app_tests_integration_documentation_CanBusManager_TestDoc.html#autotoc_md41", null ],
        [ "2. <tt>ForwardRpmDataFromMCP2515</tt>", "md_apps_cluster_app_tests_integration_documentation_CanBusManager_TestDoc.html#autotoc_md42", null ],
        [ "3. <tt>InitializeCanBusManager</tt>", "md_apps_cluster_app_tests_integration_documentation_CanBusManager_TestDoc.html#autotoc_md43", null ],
        [ "4. <tt>ManagerCleanUpBehavior</tt>", "md_apps_cluster_app_tests_integration_documentation_CanBusManager_TestDoc.html#autotoc_md44", null ]
      ] ],
      [ "Notes", "md_apps_cluster_app_tests_integration_documentation_CanBusManager_TestDoc.html#autotoc_md46", null ]
    ] ],
    [ "Integration Tests: DataManager", "md_apps_cluster_app_tests_integration_documentation_DataManager_TestDoc.html", [
      [ "Overview", "md_apps_cluster_app_tests_integration_documentation_DataManager_TestDoc.html#autotoc_md48", null ],
      [ "✅ Test Cases", "md_apps_cluster_app_tests_integration_documentation_DataManager_TestDoc.html#autotoc_md50", [
        [ "1. <tt>ForwardSpeedDataToVehicleDataManager</tt>", "md_apps_cluster_app_tests_integration_documentation_DataManager_TestDoc.html#autotoc_md51", null ],
        [ "2. <tt>ForwardRpmDataToVehicleDataManager</tt>", "md_apps_cluster_app_tests_integration_documentation_DataManager_TestDoc.html#autotoc_md52", null ],
        [ "3. <tt>ForwardSteeringDataToVehicleDataManager</tt>", "md_apps_cluster_app_tests_integration_documentation_DataManager_TestDoc.html#autotoc_md53", null ],
        [ "4. <tt>ForwardDirectionDataToVehicleDataManager</tt>", "md_apps_cluster_app_tests_integration_documentation_DataManager_TestDoc.html#autotoc_md54", null ],
        [ "5. <tt>ForwardTimeDataToSystemDataManager</tt>", "md_apps_cluster_app_tests_integration_documentation_DataManager_TestDoc.html#autotoc_md55", null ],
        [ "6. <tt>ForwardTemperatureDataToSystemDataManager</tt>", "md_apps_cluster_app_tests_integration_documentation_DataManager_TestDoc.html#autotoc_md56", null ],
        [ "7. <tt>ForwardBatteryPercentageToSystemDataManager</tt>", "md_apps_cluster_app_tests_integration_documentation_DataManager_TestDoc.html#autotoc_md57", null ],
        [ "8. <tt>ForwardMileageUpdateToVehicleDataManager</tt>", "md_apps_cluster_app_tests_integration_documentation_DataManager_TestDoc.html#autotoc_md58", null ],
        [ "9. <tt>ToggleDrivingMode</tt>", "md_apps_cluster_app_tests_integration_documentation_DataManager_TestDoc.html#autotoc_md59", null ],
        [ "10. <tt>ToggleClusterTheme</tt>", "md_apps_cluster_app_tests_integration_documentation_DataManager_TestDoc.html#autotoc_md60", null ],
        [ "11. <tt>ToggleClusterMetrics</tt>", "md_apps_cluster_app_tests_integration_documentation_DataManager_TestDoc.html#autotoc_md61", null ]
      ] ],
      [ "Notes", "md_apps_cluster_app_tests_integration_documentation_DataManager_TestDoc.html#autotoc_md63", null ]
    ] ],
    [ "Integration Tests: MCP2515Controller", "md_apps_cluster_app_tests_integration_documentation_MCP2515Device_TestDoc.html", [
      [ "Overview", "md_apps_cluster_app_tests_integration_documentation_MCP2515Device_TestDoc.html#autotoc_md65", null ],
      [ "✅ Test Cases", "md_apps_cluster_app_tests_integration_documentation_MCP2515Device_TestDoc.html#autotoc_md67", [
        [ "1. <tt>DataFrameTest</tt>", "md_apps_cluster_app_tests_integration_documentation_MCP2515Device_TestDoc.html#autotoc_md68", null ],
        [ "2. <tt>RemoteFrameTest</tt>", "md_apps_cluster_app_tests_integration_documentation_MCP2515Device_TestDoc.html#autotoc_md70", null ],
        [ "3. <tt>ErrorFrameTest</tt>", "md_apps_cluster_app_tests_integration_documentation_MCP2515Device_TestDoc.html#autotoc_md72", null ],
        [ "4. <tt>MaxBusSpeedTest</tt>", "md_apps_cluster_app_tests_integration_documentation_MCP2515Device_TestDoc.html#autotoc_md74", null ],
        [ "5. <tt>MinBusSpeedTest</tt>", "md_apps_cluster_app_tests_integration_documentation_MCP2515Device_TestDoc.html#autotoc_md76", null ]
      ] ],
      [ "Notes", "md_apps_cluster_app_tests_integration_documentation_MCP2515Device_TestDoc.html#autotoc_md78", null ]
    ] ],
    [ "Integration Tests: MileageManager", "md_apps_cluster_app_tests_integration_documentation_MileageManager_TestDoc.html", [
      [ "Overview", "md_apps_cluster_app_tests_integration_documentation_MileageManager_TestDoc.html#autotoc_md80", null ],
      [ "✅ Test Cases", "md_apps_cluster_app_tests_integration_documentation_MileageManager_TestDoc.html#autotoc_md82", [
        [ "1. <tt>ForwardMileageData</tt>", "md_apps_cluster_app_tests_integration_documentation_MileageManager_TestDoc.html#autotoc_md83", null ],
        [ "2. <tt>InitializeMileageManager</tt>", "md_apps_cluster_app_tests_integration_documentation_MileageManager_TestDoc.html#autotoc_md84", null ],
        [ "3. <tt>UpdateMileageOnSpeedUpdate</tt>", "md_apps_cluster_app_tests_integration_documentation_MileageManager_TestDoc.html#autotoc_md85", null ],
        [ "4. <tt>SaveMileage</tt>", "md_apps_cluster_app_tests_integration_documentation_MileageManager_TestDoc.html#autotoc_md86", null ],
        [ "5. <tt>UpdateTimerInterval</tt>", "md_apps_cluster_app_tests_integration_documentation_MileageManager_TestDoc.html#autotoc_md87", null ],
        [ "6. <tt>ShutdownMileageManager</tt>", "md_apps_cluster_app_tests_integration_documentation_MileageManager_TestDoc.html#autotoc_md88", null ]
      ] ],
      [ "Notes", "md_apps_cluster_app_tests_integration_documentation_MileageManager_TestDoc.html#autotoc_md90", null ]
    ] ],
    [ "Integration Tests: SystemManager", "md_apps_cluster_app_tests_integration_documentation_SystemManager_TestDoc.html", [
      [ "Overview", "md_apps_cluster_app_tests_integration_documentation_SystemManager_TestDoc.html#autotoc_md92", null ],
      [ "✅ Test Cases", "md_apps_cluster_app_tests_integration_documentation_SystemManager_TestDoc.html#autotoc_md94", [
        [ "1. <tt>UpdateTimeSignal</tt>", "md_apps_cluster_app_tests_integration_documentation_SystemManager_TestDoc.html#autotoc_md95", null ],
        [ "2. <tt>UpdateWifiStatusSignal</tt>", "md_apps_cluster_app_tests_integration_documentation_SystemManager_TestDoc.html#autotoc_md96", null ],
        [ "3. <tt>UpdateTemperatureSignal</tt>", "md_apps_cluster_app_tests_integration_documentation_SystemManager_TestDoc.html#autotoc_md97", null ],
        [ "4. <tt>UpdateBatteryPercentageSignal</tt>", "md_apps_cluster_app_tests_integration_documentation_SystemManager_TestDoc.html#autotoc_md98", null ],
        [ "5. <tt>ShutdownSystemManager</tt>", "md_apps_cluster_app_tests_integration_documentation_SystemManager_TestDoc.html#autotoc_md99", null ]
      ] ],
      [ "Notes", "md_apps_cluster_app_tests_integration_documentation_SystemManager_TestDoc.html#autotoc_md101", null ]
    ] ],
    [ "Unit Tests: CanBusManager", "md_apps_cluster_app_tests_unit_canbus_documentation_CanBusManager_TestsDoc.html", [
      [ "Overview", "md_apps_cluster_app_tests_unit_canbus_documentation_CanBusManager_TestsDoc.html#autotoc_md114", null ],
      [ "✅ Test Cases", "md_apps_cluster_app_tests_unit_canbus_documentation_CanBusManager_TestsDoc.html#autotoc_md116", [
        [ "1. <tt>SpeedSignalEmitsCorrectly</tt>", "md_apps_cluster_app_tests_unit_canbus_documentation_CanBusManager_TestsDoc.html#autotoc_md117", null ],
        [ "2. <tt>RpmSignalEmitsCorrectly</tt>", "md_apps_cluster_app_tests_unit_canbus_documentation_CanBusManager_TestsDoc.html#autotoc_md119", null ],
        [ "3. <tt>InitializeFailsWhenControllerFails</tt>", "md_apps_cluster_app_tests_unit_canbus_documentation_CanBusManager_TestsDoc.html#autotoc_md121", null ],
        [ "4. <tt>InitializeSucceedsWhenControllerSucceeds</tt>", "md_apps_cluster_app_tests_unit_canbus_documentation_CanBusManager_TestsDoc.html#autotoc_md123", null ],
        [ "5. <tt>DestructorCallsStopReading</tt>", "md_apps_cluster_app_tests_unit_canbus_documentation_CanBusManager_TestsDoc.html#autotoc_md125", null ]
      ] ],
      [ "Notes", "md_apps_cluster_app_tests_unit_canbus_documentation_CanBusManager_TestsDoc.html#autotoc_md127", null ]
    ] ],
    [ "Unit Tests: CANMessageProcessor", "md_apps_cluster_app_tests_unit_canbus_documentation_CANMessageProcessor_TestDoc.html", [
      [ "Overview", "md_apps_cluster_app_tests_unit_canbus_documentation_CANMessageProcessor_TestDoc.html#autotoc_md129", null ],
      [ "✅ Test Cases", "md_apps_cluster_app_tests_unit_canbus_documentation_CANMessageProcessor_TestDoc.html#autotoc_md131", [
        [ "1. <tt>RegisterHandlerSuccess</tt>", "md_apps_cluster_app_tests_unit_canbus_documentation_CANMessageProcessor_TestDoc.html#autotoc_md132", null ],
        [ "2. <tt>RegisterHandlerNullThrowsException</tt>", "md_apps_cluster_app_tests_unit_canbus_documentation_CANMessageProcessor_TestDoc.html#autotoc_md134", null ],
        [ "3. <tt>ProcessMessageWithRegisteredHandler</tt>", "md_apps_cluster_app_tests_unit_canbus_documentation_CANMessageProcessor_TestDoc.html#autotoc_md136", null ],
        [ "4. <tt>ProcessMessageWithUnregisteredHandlerThrowsException</tt>", "md_apps_cluster_app_tests_unit_canbus_documentation_CANMessageProcessor_TestDoc.html#autotoc_md138", null ],
        [ "5. <tt>OverwriteHandlerForSameFrameID</tt>", "md_apps_cluster_app_tests_unit_canbus_documentation_CANMessageProcessor_TestDoc.html#autotoc_md140", null ]
      ] ],
      [ "Notes", "md_apps_cluster_app_tests_unit_canbus_documentation_CANMessageProcessor_TestDoc.html#autotoc_md142", null ]
    ] ],
    [ "Unit Tests: MCP2515Configurator", "md_apps_cluster_app_tests_unit_canbus_documentation_MCP2515Configurator_TestDoc.html", [
      [ "Overview", "md_apps_cluster_app_tests_unit_canbus_documentation_MCP2515Configurator_TestDoc.html#autotoc_md144", null ],
      [ "✅ Test Cases", "md_apps_cluster_app_tests_unit_canbus_documentation_MCP2515Configurator_TestDoc.html#autotoc_md146", [
        [ "1. <tt>ResetChipSuccess</tt>", "md_apps_cluster_app_tests_unit_canbus_documentation_MCP2515Configurator_TestDoc.html#autotoc_md147", null ],
        [ "2. <tt>ResetChipFailure</tt>", "md_apps_cluster_app_tests_unit_canbus_documentation_MCP2515Configurator_TestDoc.html#autotoc_md149", null ],
        [ "3. <tt>ConfigureBaudRate</tt>", "md_apps_cluster_app_tests_unit_canbus_documentation_MCP2515Configurator_TestDoc.html#autotoc_md151", null ],
        [ "4. <tt>ConfigureTXBuffer</tt>", "md_apps_cluster_app_tests_unit_canbus_documentation_MCP2515Configurator_TestDoc.html#autotoc_md153", null ],
        [ "5. <tt>ConfigureRXBuffer</tt>", "md_apps_cluster_app_tests_unit_canbus_documentation_MCP2515Configurator_TestDoc.html#autotoc_md155", null ],
        [ "6. <tt>ConfigureFiltersAndMasks</tt>", "md_apps_cluster_app_tests_unit_canbus_documentation_MCP2515Configurator_TestDoc.html#autotoc_md157", null ],
        [ "7. <tt>ConfigureInterrupts</tt>", "md_apps_cluster_app_tests_unit_canbus_documentation_MCP2515Configurator_TestDoc.html#autotoc_md159", null ],
        [ "8. <tt>SetMode</tt>", "md_apps_cluster_app_tests_unit_canbus_documentation_MCP2515Configurator_TestDoc.html#autotoc_md161", null ],
        [ "9. <tt>VerifyModeSuccess</tt>", "md_apps_cluster_app_tests_unit_canbus_documentation_MCP2515Configurator_TestDoc.html#autotoc_md163", null ],
        [ "10. <tt>VerifyModeFailure</tt>", "md_apps_cluster_app_tests_unit_canbus_documentation_MCP2515Configurator_TestDoc.html#autotoc_md165", null ],
        [ "11. <tt>ReadCANMessageWithData</tt>", "md_apps_cluster_app_tests_unit_canbus_documentation_MCP2515Configurator_TestDoc.html#autotoc_md167", null ],
        [ "12. <tt>ReadCANMessageNoData</tt>", "md_apps_cluster_app_tests_unit_canbus_documentation_MCP2515Configurator_TestDoc.html#autotoc_md169", null ]
      ] ],
      [ "Notes", "md_apps_cluster_app_tests_unit_canbus_documentation_MCP2515Configurator_TestDoc.html#autotoc_md171", null ]
    ] ],
    [ "Unit Tests: MCP2515Controller", "md_apps_cluster_app_tests_unit_canbus_documentation_MCP2515Controller_TestDoc.html", [
      [ "Overview", "md_apps_cluster_app_tests_unit_canbus_documentation_MCP2515Controller_TestDoc.html#autotoc_md173", null ],
      [ "✅ Test Cases", "md_apps_cluster_app_tests_unit_canbus_documentation_MCP2515Controller_TestDoc.html#autotoc_md175", [
        [ "1. <tt>InitializationSuccess</tt>", "md_apps_cluster_app_tests_unit_canbus_documentation_MCP2515Controller_TestDoc.html#autotoc_md176", null ],
        [ "2. <tt>InitializationFailure</tt>", "md_apps_cluster_app_tests_unit_canbus_documentation_MCP2515Controller_TestDoc.html#autotoc_md178", null ],
        [ "3. <tt>SetupHandlersTest</tt>", "md_apps_cluster_app_tests_unit_canbus_documentation_MCP2515Controller_TestDoc.html#autotoc_md180", null ],
        [ "4. <tt>SpeedUpdatedSignal</tt>", "md_apps_cluster_app_tests_unit_canbus_documentation_MCP2515Controller_TestDoc.html#autotoc_md182", null ],
        [ "5. <tt>RpmUpdatedSignal</tt>", "md_apps_cluster_app_tests_unit_canbus_documentation_MCP2515Controller_TestDoc.html#autotoc_md184", null ],
        [ "6. <tt>ProcessReadingCallsHandlers</tt>", "md_apps_cluster_app_tests_unit_canbus_documentation_MCP2515Controller_TestDoc.html#autotoc_md186", null ],
        [ "7. <tt>StopReadingStopsProcessing</tt>", "md_apps_cluster_app_tests_unit_canbus_documentation_MCP2515Controller_TestDoc.html#autotoc_md188", null ]
      ] ],
      [ "Notes", "md_apps_cluster_app_tests_unit_canbus_documentation_MCP2515Controller_TestDoc.html#autotoc_md190", null ]
    ] ],
    [ "Unit Tests: SPIController", "md_apps_cluster_app_tests_unit_canbus_documentation_SPIController_TestDoc.html", [
      [ "Overview", "md_apps_cluster_app_tests_unit_canbus_documentation_SPIController_TestDoc.html#autotoc_md192", null ],
      [ "✅ Test Cases", "md_apps_cluster_app_tests_unit_canbus_documentation_SPIController_TestDoc.html#autotoc_md194", [
        [ "1. <tt>OpenDeviceSuccess</tt>", "md_apps_cluster_app_tests_unit_canbus_documentation_SPIController_TestDoc.html#autotoc_md195", null ],
        [ "2. <tt>OpenDeviceFailure</tt>", "md_apps_cluster_app_tests_unit_canbus_documentation_SPIController_TestDoc.html#autotoc_md197", null ],
        [ "3. <tt>ConfigureSPIValidParameters</tt>", "md_apps_cluster_app_tests_unit_canbus_documentation_SPIController_TestDoc.html#autotoc_md199", null ],
        [ "4. <tt>WriteByteSuccess</tt>", "md_apps_cluster_app_tests_unit_canbus_documentation_SPIController_TestDoc.html#autotoc_md201", null ],
        [ "5. <tt>ReadByteSuccess</tt>", "md_apps_cluster_app_tests_unit_canbus_documentation_SPIController_TestDoc.html#autotoc_md203", null ],
        [ "6. <tt>SpiTransferSuccess</tt>", "md_apps_cluster_app_tests_unit_canbus_documentation_SPIController_TestDoc.html#autotoc_md205", null ],
        [ "7. <tt>CloseDeviceSuccess</tt>", "md_apps_cluster_app_tests_unit_canbus_documentation_SPIController_TestDoc.html#autotoc_md207", null ]
      ] ],
      [ "Notes", "md_apps_cluster_app_tests_unit_canbus_documentation_SPIController_TestDoc.html#autotoc_md209", null ]
    ] ],
    [ "Unit Tests: ClusterSettingsManager", "md_apps_cluster_app_tests_unit_data_documentation_ClusterSettiingsManager_TestDoc.html", [
      [ "Overview", "md_apps_cluster_app_tests_unit_data_documentation_ClusterSettiingsManager_TestDoc.html#autotoc_md211", null ],
      [ "✅ Test Cases", "md_apps_cluster_app_tests_unit_data_documentation_ClusterSettiingsManager_TestDoc.html#autotoc_md213", [
        [ "1. <tt>ToggleDrivingModeEmitsSignal</tt>", "md_apps_cluster_app_tests_unit_data_documentation_ClusterSettiingsManager_TestDoc.html#autotoc_md214", null ],
        [ "2. <tt>ToggleClusterThemeEmitsSignal</tt>", "md_apps_cluster_app_tests_unit_data_documentation_ClusterSettiingsManager_TestDoc.html#autotoc_md216", null ],
        [ "3. <tt>ToggleClusterMetricsEmitsSignal</tt>", "md_apps_cluster_app_tests_unit_data_documentation_ClusterSettiingsManager_TestDoc.html#autotoc_md218", null ]
      ] ],
      [ "Notes", "md_apps_cluster_app_tests_unit_data_documentation_ClusterSettiingsManager_TestDoc.html#autotoc_md220", null ]
    ] ],
    [ "Unit Tests: SystemDataManager", "md_apps_cluster_app_tests_unit_data_documentation_SystemDataManager_TestDoc.html", [
      [ "Overview", "md_apps_cluster_app_tests_unit_data_documentation_SystemDataManager_TestDoc.html#autotoc_md222", null ],
      [ "✅ Test Cases", "md_apps_cluster_app_tests_unit_data_documentation_SystemDataManager_TestDoc.html#autotoc_md224", [
        [ "1. <tt>TimeDataEmitsSignal</tt>", "md_apps_cluster_app_tests_unit_data_documentation_SystemDataManager_TestDoc.html#autotoc_md225", null ],
        [ "2. <tt>TemperatureDataEmitsSignalOnChange</tt>", "md_apps_cluster_app_tests_unit_data_documentation_SystemDataManager_TestDoc.html#autotoc_md227", null ],
        [ "3. <tt>BatteryPercentageEmitsSignalOnChange</tt>", "md_apps_cluster_app_tests_unit_data_documentation_SystemDataManager_TestDoc.html#autotoc_md229", null ]
      ] ],
      [ "Notes", "md_apps_cluster_app_tests_unit_data_documentation_SystemDataManager_TestDoc.html#autotoc_md231", null ]
    ] ],
    [ "Unit Tests: VehicleDataManager", "md_apps_cluster_app_tests_unit_data_documentation_VehicleDataManager_TestDoc.html", [
      [ "Overview", "md_apps_cluster_app_tests_unit_data_documentation_VehicleDataManager_TestDoc.html#autotoc_md233", null ],
      [ "✅ Test Cases", "md_apps_cluster_app_tests_unit_data_documentation_VehicleDataManager_TestDoc.html#autotoc_md235", [
        [ "1. <tt>RpmDataEmitsSignal</tt>", "md_apps_cluster_app_tests_unit_data_documentation_VehicleDataManager_TestDoc.html#autotoc_md236", null ],
        [ "2. <tt>SpeedDataEmitsSignalInKilometers</tt>", "md_apps_cluster_app_tests_unit_data_documentation_VehicleDataManager_TestDoc.html#autotoc_md238", null ],
        [ "3. <tt>MileageDataEmitsSignalOnChange</tt>", "md_apps_cluster_app_tests_unit_data_documentation_VehicleDataManager_TestDoc.html#autotoc_md240", null ],
        [ "4. <tt>DirectionDataEmitsSignalOnChange</tt>", "md_apps_cluster_app_tests_unit_data_documentation_VehicleDataManager_TestDoc.html#autotoc_md242", null ],
        [ "5. <tt>SteeringDataEmitsSignalOnChange</tt>", "md_apps_cluster_app_tests_unit_data_documentation_VehicleDataManager_TestDoc.html#autotoc_md244", null ]
      ] ],
      [ "Notes", "md_apps_cluster_app_tests_unit_data_documentation_VehicleDataManager_TestDoc.html#autotoc_md246", null ]
    ] ],
    [ "Unit Tests: MileageCalculator", "md_apps_cluster_app_tests_unit_mileage_documentation_MileageCalculator_TestDoc.html", [
      [ "Overview", "md_apps_cluster_app_tests_unit_mileage_documentation_MileageCalculator_TestDoc.html#autotoc_md248", null ],
      [ "✅ Test Cases", "md_apps_cluster_app_tests_unit_mileage_documentation_MileageCalculator_TestDoc.html#autotoc_md250", [
        [ "1. <tt>AddSpeed_DoesNotCrash</tt>", "md_apps_cluster_app_tests_unit_mileage_documentation_MileageCalculator_TestDoc.html#autotoc_md251", null ],
        [ "2. <tt>CalculateDistance_NoSpeeds_ReturnsZero</tt>", "md_apps_cluster_app_tests_unit_mileage_documentation_MileageCalculator_TestDoc.html#autotoc_md253", null ],
        [ "3. <tt>CalculateDistance_ZeroSpeed_ReturnsZero</tt>", "md_apps_cluster_app_tests_unit_mileage_documentation_MileageCalculator_TestDoc.html#autotoc_md255", null ],
        [ "4. <tt>CalculateDistance_BasicCalculation</tt>", "md_apps_cluster_app_tests_unit_mileage_documentation_MileageCalculator_TestDoc.html#autotoc_md257", null ],
        [ "5. <tt>CalculateDistance_MultipleSpeeds</tt>", "md_apps_cluster_app_tests_unit_mileage_documentation_MileageCalculator_TestDoc.html#autotoc_md259", null ]
      ] ],
      [ "Notes", "md_apps_cluster_app_tests_unit_mileage_documentation_MileageCalculator_TestDoc.html#autotoc_md261", null ]
    ] ],
    [ "Unit Tests: MileageFileHandler", "md_apps_cluster_app_tests_unit_mileage_documentation_MileageFileHandler_TestDoc.html", [
      [ "Overview", "md_apps_cluster_app_tests_unit_mileage_documentation_MileageFileHandler_TestDoc.html#autotoc_md263", null ],
      [ "✅ Test Cases", "md_apps_cluster_app_tests_unit_mileage_documentation_MileageFileHandler_TestDoc.html#autotoc_md265", [
        [ "1. <tt>EnsureFileExists_FileDoesNotExist_CreatesFileWithZeroMileage</tt>", "md_apps_cluster_app_tests_unit_mileage_documentation_MileageFileHandler_TestDoc.html#autotoc_md266", null ],
        [ "2. <tt>ReadMileage_ValidNumber_ReturnsParsedValue</tt>", "md_apps_cluster_app_tests_unit_mileage_documentation_MileageFileHandler_TestDoc.html#autotoc_md268", null ],
        [ "3. <tt>ReadMileage_InvalidNumber_ReturnsZero</tt>", "md_apps_cluster_app_tests_unit_mileage_documentation_MileageFileHandler_TestDoc.html#autotoc_md270", null ],
        [ "4. <tt>WriteMileage_ValidNumber_WritesToFile</tt>", "md_apps_cluster_app_tests_unit_mileage_documentation_MileageFileHandler_TestDoc.html#autotoc_md272", null ]
      ] ],
      [ "Notes", "md_apps_cluster_app_tests_unit_mileage_documentation_MileageFileHandler_TestDoc.html#autotoc_md274", null ]
    ] ],
    [ "Unit Tests: MileageManager", "md_apps_cluster_app_tests_unit_mileage_documentation_MileageManager_TestDoc.html", [
      [ "Overview", "md_apps_cluster_app_tests_unit_mileage_documentation_MileageManager_TestDoc.html#autotoc_md276", null ],
      [ "✅ Test Cases", "md_apps_cluster_app_tests_unit_mileage_documentation_MileageManager_TestDoc.html#autotoc_md278", [
        [ "1. <tt>Initialize_LoadsMileageFromFile</tt>", "md_apps_cluster_app_tests_unit_mileage_documentation_MileageManager_TestDoc.html#autotoc_md279", null ],
        [ "2. <tt>OnSpeedUpdated_CallsCalculator</tt>", "md_apps_cluster_app_tests_unit_mileage_documentation_MileageManager_TestDoc.html#autotoc_md281", null ],
        [ "3. <tt>UpdateMileage_EmitsMileageUpdatedSignal</tt>", "md_apps_cluster_app_tests_unit_mileage_documentation_MileageManager_TestDoc.html#autotoc_md283", null ],
        [ "4. <tt>SaveMileage_CallsFileHandler</tt>", "md_apps_cluster_app_tests_unit_mileage_documentation_MileageManager_TestDoc.html#autotoc_md285", null ]
      ] ],
      [ "Notes", "md_apps_cluster_app_tests_unit_mileage_documentation_MileageManager_TestDoc.html#autotoc_md287", null ]
    ] ],
    [ "Unit Tests: BatteryController", "md_apps_cluster_app_tests_unit_system_documentation_BatteryController_TestDoc.html", [
      [ "Overview", "md_apps_cluster_app_tests_unit_system_documentation_BatteryController_TestDoc.html#autotoc_md289", null ],
      [ "✅ Test Cases", "md_apps_cluster_app_tests_unit_system_documentation_BatteryController_TestDoc.html#autotoc_md291", [
        [ "1. <tt>Initialization_CallsCalibration</tt>", "md_apps_cluster_app_tests_unit_system_documentation_BatteryController_TestDoc.html#autotoc_md292", null ],
        [ "2. <tt>GetBatteryPercentage_CorrectCalculation</tt>", "md_apps_cluster_app_tests_unit_system_documentation_BatteryController_TestDoc.html#autotoc_md294", null ]
      ] ],
      [ "Notes", "md_apps_cluster_app_tests_unit_system_documentation_BatteryController_TestDoc.html#autotoc_md296", null ]
    ] ],
    [ "Unit Tests: SystemInfoProvider", "md_apps_cluster_app_tests_unit_system_documentation_SystemInfoProvider_TestDoc.html", [
      [ "Overview", "md_apps_cluster_app_tests_unit_system_documentation_SystemInfoProvider_TestDoc.html#autotoc_md298", null ],
      [ "✅ Test Cases", "md_apps_cluster_app_tests_unit_system_documentation_SystemInfoProvider_TestDoc.html#autotoc_md300", [
        [ "1. <tt>GetWifiStatus_Connected</tt>", "md_apps_cluster_app_tests_unit_system_documentation_SystemInfoProvider_TestDoc.html#autotoc_md301", null ],
        [ "2. <tt>GetWifiStatus_Disconnected</tt>", "md_apps_cluster_app_tests_unit_system_documentation_SystemInfoProvider_TestDoc.html#autotoc_md303", null ],
        [ "3. <tt>GetWifiStatus_NoInterface</tt>", "md_apps_cluster_app_tests_unit_system_documentation_SystemInfoProvider_TestDoc.html#autotoc_md305", null ],
        [ "4. <tt>GetTemperature_ValidReading</tt>", "md_apps_cluster_app_tests_unit_system_documentation_SystemInfoProvider_TestDoc.html#autotoc_md307", null ],
        [ "5. <tt>GetTemperature_InvalidReading</tt>", "md_apps_cluster_app_tests_unit_system_documentation_SystemInfoProvider_TestDoc.html#autotoc_md309", null ],
        [ "6. <tt>GetIpAddress_Valid</tt>", "md_apps_cluster_app_tests_unit_system_documentation_SystemInfoProvider_TestDoc.html#autotoc_md311", null ],
        [ "7. <tt>GetIpAddress_NoIP</tt>", "md_apps_cluster_app_tests_unit_system_documentation_SystemInfoProvider_TestDoc.html#autotoc_md313", null ]
      ] ],
      [ "Notes", "md_apps_cluster_app_tests_unit_system_documentation_SystemInfoProvider_TestDoc.html#autotoc_md315", null ]
    ] ],
    [ "Unit Tests: SystemManager", "md_apps_cluster_app_tests_unit_system_documentation_SystemManager_TestDoc.html", [
      [ "Overview", "md_apps_cluster_app_tests_unit_system_documentation_SystemManager_TestDoc.html#autotoc_md317", null ],
      [ "✅ Test Cases", "md_apps_cluster_app_tests_unit_system_documentation_SystemManager_TestDoc.html#autotoc_md319", [
        [ "1. <tt>UpdateTime_EmitsCorrectSignal</tt>", "md_apps_cluster_app_tests_unit_system_documentation_SystemManager_TestDoc.html#autotoc_md320", null ],
        [ "2. <tt>UpdateSystemStatus_EmitsWifiStatus</tt>", "md_apps_cluster_app_tests_unit_system_documentation_SystemManager_TestDoc.html#autotoc_md322", null ],
        [ "3. <tt>UpdateSystemStatus_EmitsTemperature</tt>", "md_apps_cluster_app_tests_unit_system_documentation_SystemManager_TestDoc.html#autotoc_md324", null ],
        [ "4. <tt>UpdateSystemStatus_EmitsBatteryPercentage</tt>", "md_apps_cluster_app_tests_unit_system_documentation_SystemManager_TestDoc.html#autotoc_md326", null ]
      ] ],
      [ "Notes", "md_apps_cluster_app_tests_unit_system_documentation_SystemManager_TestDoc.html#autotoc_md328", null ]
    ] ],
    [ "HotWheels - Instrument Cluster ⏲ 🎮", "md_Instrument_Cluster.html", [
      [ "Module Description", "md_Instrument_Cluster.html#autotoc_md341", null ],
      [ "Communication Architecture", "md_Instrument_Cluster.html#autotoc_md342", null ],
      [ "Software Architecture", "md_Instrument_Cluster.html#autotoc_md343", null ],
      [ "Adaptation to Read CAN Messages Using SPI Pins", "md_Instrument_Cluster.html#autotoc_md344", null ],
      [ "Cross-Compilation Method", "md_Instrument_Cluster.html#autotoc_md345", null ],
      [ "Results", "md_Instrument_Cluster.html#autotoc_md346", null ]
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
"car__controls_2sources_2main_8cpp.html#a0ddf1224851353fc92bfbff6f499fa97",
"classIInferencer.html#ac04789c02b26e8357a00e9931654813d",
"classNotificationOverlay.html",
"cluster_2includes_2data_2enums_8hpp.html#a7742add5e28549b88107eeec57b2f835ae599161956d626eda4cb0a5ffb85271c",
"md_apps_cluster_app_tests_integration_documentation_SystemManager_TestDoc.html#autotoc_md95",
"test__MCP2515Device_8cpp.html#adeb98d55d2e51e87afc2965bf98bd9b1"
];

var SYNCONMSG = 'click to disable panel synchronisation';
var SYNCOFFMSG = 'click to enable panel synchronisation';