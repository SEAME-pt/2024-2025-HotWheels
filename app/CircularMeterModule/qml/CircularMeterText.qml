import QtQuick 2.15

Item {
    id: meterText
    anchors.fill: parent

    property real displayedValue: 0

    Text {
        id: meterValue
        font.pixelSize: meterFontSize
        color: "white"
        text: Math.round(meterText.displayedValue)
        anchors.horizontalCenter: parent.horizontalCenter
        anchors.verticalCenter: parent.verticalCenter
    }

    Text {
        id: meterUnit
        font.pixelSize: 24
        color: "white"
        text: meterLabel
        anchors.horizontalCenter: meterValue.horizontalCenter
        anchors.top: meterValue.bottom
    }
}
