#ifndef COMMON_TYPES_HPP
#define COMMON_TYPES_HPP

struct Point2D {
		double x, y;
		Point2D (double x_ = 0.0, double y_ = 0.0) : x (x_), y (y_) {}
};

struct ControlCommand {
		int throttle, steer;

		ControlCommand () : throttle (0), steer (0) {}
		ControlCommand (int t, int s) : throttle (t), steer (s) {}
    };

#endif