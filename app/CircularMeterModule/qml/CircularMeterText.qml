import QtQuick 2.15

Item {
    id: meterText
    anchors.fill: parent

    property real displayedValue: 0

    Text {
        id: meterValue
        font.pixelSize: Math.min(parent.width, parent.height) * 0.2
        color: "white"
        text: Math.round(meterText.displayedValue)
        anchors.horizontalCenter: parent.horizontalCenter
        anchors.verticalCenter: parent.verticalCenter
    }

    Text {
        id: meterUnit
        font.pixelSize: Math.min(parent.width, parent.height) * 0.1
        color: "white"
        text: meterLabel
        anchors.horizontalCenter: meterValue.horizontalCenter
        anchors.top: meterValue.bottom
    }

}
