import QtQuick 2.15
import "CircularMeterUtils.js" as Utils

Rectangle {
    id: circularMeter
    width: 300
    height: 300
    color: parentColor
    anchors.centerIn: parent

    property real displayedValue: 0

    OuterCircle { anchors.fill: parent }
    CircularMeterArc {
        anchors.fill: parent
        displayedValue: circularMeter.displayedValue
    }
    CircleTicks { anchors.fill: parent }
    CircularMeterText {
        anchors.fill: parent
        displayedValue: circularMeter.displayedValue
    }

    Connections {
        target: meterController
        function onValueChanged() {
            if (animation.running) {
                animation.stop(); // Stop the animation if it's already running
            }
            animation.to = meterController.value; // Update the target value
            animation.start(); // Restart the animation
        }
    }

    NumberAnimation {
        id: animation
        target: circularMeter
        property: "displayedValue"
        duration: 200
        easing.type: Easing.InOutQuad
    }
}
