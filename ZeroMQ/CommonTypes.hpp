#ifndef COMMON_TYPES_HPP
#define COMMON_TYPES_HPP

struct Point2D
{
    double x, y;
    Point2D(double x_ = 0.0, double y_ = 0.0) : x(x_), y(y_) {}
};

struct ControlCommand
{
    int throttle, steer;

    ControlCommand() : throttle(0), steer(0) {}
    ControlCommand(int t, int s) : throttle(t), steer(s) {}
};

struct LaneCurve
{
    std::vector<Point2D> centroids;
    std::vector<Point2D> curve;
};

struct CenterlineResult
{
    std::vector<Point2D> blended;
    std::vector<Point2D> midpoint;
    std::vector<Point2D> straight;
    std::vector<LaneCurve> lanes;
    bool valid;

    CenterlineResult() : valid(false) {}
};

// --- Publisher Port Constants ---

const std::string CAR_SPEED_TOPIC = "car_speed";
const int CAR_SPEED_PORT = 5568;

const std::string OBJECT_TOPIC = "notification";
const int OBJECT_PORT = 5557;

const std::string INFERENCE_TOPIC = "inference_frame";
const int INFERENCE_PORT = 5556;

const std::string POLYFITTING_TOPIC = "polyfitting_result";
const int POLYFITTING_PORT = 5569;

static const std::string getZeroMQAddress(int port) {
    return "tcp://localhost:" + std::to_string(port);
}

#endif