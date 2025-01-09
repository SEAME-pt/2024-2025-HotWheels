import QtQuick 2.15
import "CircularMeterUtils.js" as Utils

Canvas {
    id: meterArc
    anchors.fill: parent

    property real displayedValue: 0

    onPaint: {
        var ctx = getContext("2d");
        ctx.reset();

        var centerX = width / 2;
        var centerY = height / 2;
        var radius = Math.min(width, height) / 2 - width / 20;
        Utils.drawBackgroundCircle(ctx, centerX, centerY, radius);
        Utils.drawCircularMeter(ctx, centerX, centerY, radius, displayedValue, meterController.maxValue);
    }

    Connections {
        target: circularMeter
        function onDisplayedValueChanged() {meterArc.requestPaint()}
    }
}
