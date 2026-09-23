#ifndef LATENCY_LOGGER_HPP
#define LATENCY_LOGGER_HPP

#include <vector>
#include <chrono>
#include <Eigen/Dense>
#include <iostream>

class LatencyLogger {
public:
    LatencyLogger(int num_motors, int buffer_size = 1000)
        : num_motors_(num_motors), buffer_size_(buffer_size) {
        latency_buffer_.resize(num_motors, buffer_size);
        latency_buffer_.setZero();
        
        latency_idx_.resize(num_motors);
        latency_idx_.setZero();
        
        last_send_time_.resize(num_motors);
        last_send_time_.setZero();
    }

    // Record the time a command was sent
    void log_start(int motor_id, double timestamp) {
        if (motor_id >= 0 && motor_id < num_motors_) {
            last_send_time_(motor_id) = timestamp;
        }
    }

    // Record the time a response was received and compute latency
    // Record time with kernel timestamp option
    // Record the time a response was received and compute latency
    void log_end(int motor_id, double timestamp) {
        if (motor_id >= 0 && motor_id < num_motors_) {
            double start_time = last_send_time_(motor_id);
            double diff = timestamp - start_time;

            if (diff > 0 && diff < 1.0) { // Simple validity check (0 to 1 second)
                // Store in ms
                latency_buffer_(motor_id, latency_idx_(motor_id)) = (float)(diff * 1000.0);
            } else {
                latency_buffer_(motor_id, latency_idx_(motor_id)) = 0.0f;
            }

            // Update circular buffer index
            latency_idx_(motor_id) = (latency_idx_(motor_id) + 1) % buffer_size_;
        }
    }

    // Retrieve history for a specific motor
    std::vector<float> get_history(int motor_id) const {
        if (motor_id < 0 || motor_id >= num_motors_) {
            return {};
        }
        Eigen::VectorXf row = latency_buffer_.row(motor_id);
        return std::vector<float>(row.data(), row.data() + row.size());
    }

private:
    int num_motors_;
    int buffer_size_;

    // Rows: motors, Cols: history buffer
    Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic> latency_buffer_;
    Eigen::VectorXi latency_idx_;
    Eigen::VectorXd last_send_time_;
};

#endif // LATENCY_LOGGER_HPP
