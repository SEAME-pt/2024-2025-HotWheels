import QtQuick 2.15

Rectangle {
    id: outerCircle
    width: parent.width - 18
    height: parent.height - 18
    anchors.centerIn: parent
    color: "#36454F"
    radius: width / 2
    border.color: "#1E90FF"
    border.width: 8
}
