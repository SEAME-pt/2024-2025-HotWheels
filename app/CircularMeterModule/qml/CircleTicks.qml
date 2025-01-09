import QtQuick 2.15
import "CircularMeterUtils.js" as Utils

Canvas {
    id: tickCanvas
    anchors.fill: parent

    onPaint: {
        var ctx = getContext("2d");
        ctx.reset();

        var centerX = width / 2;
        var centerY = height / 2;
        var radius = width / 2 - 5;

        Utils.drawTicks(ctx, centerX, centerY, radius, meterController.maxValue, { ticksInterval: ticksInterval });
    }
}
