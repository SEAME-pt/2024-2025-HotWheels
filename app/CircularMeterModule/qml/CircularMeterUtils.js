
function drawBackgroundCircle(ctx, centerX, centerY, radius) {
    ctx.beginPath();
    ctx.arc(centerX, centerY, radius, 0, Math.PI * 2, false);
    ctx.lineWidth = 28;
    ctx.strokeStyle = "#000000";
    ctx.stroke();
}

function drawCircularMeter(ctx, centerX, centerY, radius, value, maxValue) {
    var angleStart = Math.PI * 0.8;
    var angleRange = Math.PI * 1.4;
    var angle = angleStart + angleRange * (value / maxValue);

    ctx.beginPath();
    ctx.arc(centerX, centerY, radius, angle - 0.02, angle + 0.02, false);
    ctx.lineWidth = 28;
    ctx.strokeStyle = "white";
    ctx.stroke();
}

function drawTicks(ctx, centerX, centerY, radius, maxValue, options = {}) {
    var angleStart = Math.PI * 0.8;
    var angleRange = Math.PI * 1.4;
    var totalTicks = maxValue / options.ticksInterval || 20;

    var tickLengthMajor = options.tickLengthMajor || 15;
    var tickLengthMinor = options.tickLengthMinor || 8;
    var tickWidthMajor = options.tickWidthMajor || 3;
    var tickWidthMinor = options.tickWidthMinor || 1.5;
    var labelOffset = options.labelOffset || 20;
    var majorTickColor = options.majorTickColor || "white";
    var minorTickColor = options.minorTickColor || "gray";
    var labelColor = options.labelColor || "white";

    for (var i = 0; i <= totalTicks; i++) {
        var angle = angleStart + (i / totalTicks) * angleRange;
        var isMajor = i % 2 === 0;

        var tickLength = isMajor ? tickLengthMajor : tickLengthMinor;
        var tickWidth = isMajor ? tickWidthMajor : tickWidthMinor;

        ctx.beginPath();
        ctx.lineWidth = tickWidth;
        ctx.strokeStyle = isMajor ? majorTickColor : minorTickColor;

        ctx.moveTo(
            centerX + Math.cos(angle) * (radius - tickLength),
            centerY + Math.sin(angle) * (radius - tickLength)
        );
        ctx.lineTo(
            centerX + Math.cos(angle) * radius,
            centerY + Math.sin(angle) * radius
        );
        ctx.stroke();

        if (isMajor) {
            var labelRadius = radius - tickLength - labelOffset;
            var label = i * (options.ticksInterval || 20);
            ctx.fillStyle = labelColor;
            ctx.textAlign = "center";
            ctx.textBaseline = "middle";
            ctx.fillText(
                label.toString(),
                centerX + Math.cos(angle) * labelRadius,
                centerY + Math.sin(angle) * labelRadius
            );
        }
    }
}



