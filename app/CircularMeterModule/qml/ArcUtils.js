function paintArc(ctx, root) {
    var x = root.width / 2;
    var y = root.height / 2;
    var start = Math.PI * (root.arcBegin / 180);
    var end = Math.PI * (root.arcEnd / 180);
    ctx.reset();

    if (root.isPie) {
        if (root.showBackground) {
            ctx.beginPath();
            ctx.fillStyle = root.colorBackground;
            ctx.moveTo(x, y);
            ctx.arc(x, y, root.width / 2, 0, Math.PI * 2, false);
            ctx.lineTo(x, y);
            ctx.fill();
        }
        ctx.beginPath();
        ctx.fillStyle = root.colorCircle;
        ctx.moveTo(x, y);
        ctx.arc(x, y, root.width / 2, start, end, false);
        ctx.lineTo(x, y);
        ctx.fill();
    } else {
        if (root.showBackground) {
            ctx.beginPath();
            ctx.arc(x, y, (root.width / 2) - root.lineWidth / 2, 0, Math.PI * 2, false);
            ctx.lineWidth = root.lineWidth;
            ctx.strokeStyle = root.colorBackground;
            ctx.stroke();
        }
        ctx.beginPath();
        ctx.arc(x, y, (root.width / 2) - root.lineWidth / 2, start, end, false);
        ctx.lineWidth = root.lineWidth;
        ctx.strokeStyle = root.colorCircle;
        ctx.stroke();
    }
}
