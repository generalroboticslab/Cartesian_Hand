#ifndef __MOTOR_HPP__
#define __MOTOR_HPP__

#include <iostream>
#include <cstring>
#include <cerrno>
#include <sys/types.h>
#include <sys/socket.h>
#include <linux/can.h>
#include <linux/can/raw.h>
#include <net/if.h>
#include <unistd.h>
#include <fcntl.h>     // For fcntl and O_NONBLOCK
#include <sys/ioctl.h> // For ioctl and SIOCGIFINDEX
#include <net/if.h>    // For struct ifreq and IFNAMSIZ
#include <poll.h>      // For poll
#include <cstdint>
#include <unistd.h>
#include <time.h>      // for timespec, clock_gettime, clock_nanosleep
#include <sched.h>     // for sched_setscheduler, SCHED_FIFO
#include <sys/mman.h>  // for mlockall, MCL_CURRENT, MCL_FUTURE

#include <thread>
#include <queue>
#include <mutex>
#include <condition_variable>
#include <atomic>
#include "latency_logger.hpp"

#include <typeinfo> // Include for typeid

#include <cassert>

#include <Eigen/Dense>
#include <map>

#include <fmt/core.h>
#include <fmt/chrono.h>
#include <fmt/color.h>

enum MotorCommunicationType : uint8_t
{
    GetCanID = 0x00,           // Get device ID and 64-bit MCU unique identifier
    MotionControl = 0x01,      // Motion control mode for sending control commands to the host
    MotorStatus = 0x02,        // Used to feedback motor running status to the host
    Enable = 0x03,             // Enable motor operation
    Disable = 0x04,            // Stop motor operation
    SetPosZero = 0x06,         // Set motor mechanical zero position
    ChangeCanID = 0x07,        // Change the current motor CAN ID
    GetSingleParameter = 0x11, // Read single parameter
    SetSingleParameter = 0x12, // Set single parameter
    ControlMode = 0x12,        // Set motor mode
    Error = 0x15,              // Fault feedback frame
    SaveData = 0x16,           // Motor data save frame
};

// template for motor specification
struct MotorSpec
{
    float pos_min;
    float pos_max;
    float vel_min; // check the manual communication type 2 mtor feedback data
    float vel_max; // check the manual communication type 2 mtor feedback data
    float kp_min; // TODO VERIFY THIS
    float kp_max;
    float kd_min;
    float kd_max;
    float torque_min;
    float torque_max;
};
// refer to the manual communication type 2
constexpr MotorSpec motor_spec_r01 = {-12.57f, 12.57f, -44.0f, 44.0f, 0.0f, 500.00f, 0.0f, 5.000f, -17.00f, 17.00f};
constexpr MotorSpec motor_spec_r02 = {-12.57f, 12.57f, -44.0f, 44.0f, 0.0f, 500.00f, 0.0f, 5.000f, -17.00f, 17.00f};
constexpr MotorSpec motor_spec_r03 = {-12.57f, 12.57f, -20.0f, 20.0f, 0.0f, 5000.0f, 0.0f, 100.0f, -60.00f, 60.00f};
constexpr MotorSpec motor_spec_r04 = {-12.57f, 12.57f, -15.0f, 15.0f, 0.0f, 5000.0f, 0.0f, 100.0f, -120.0f, 120.0f};
constexpr MotorSpec motor_spec_r05 = {-12.57f, 12.57f, -50.0f, 50.0f, 0.0f, 500.00f, 0.0f, 5.000f, -5.500f, 5.500f};
constexpr MotorSpec motor_spec_r06 = {-12.57f, 12.57f, -50.0f, 50.0f, 0.0f, 5000.0f, 0.0f, 100.0f, -36.00f, 36.00f};
constexpr MotorSpec motor_spec_r00 = {-12.57f, 12.57f, -33.0f, 33.0f, 0.0f, 500.00f, 0.0f, 5.000f, -14.00f, 14.00f};



enum MotorType : uint8_t
{
    R00 = 0,
    R01 = 1,
    R02 = 2,
    R03 = 3,
    R04 = 4,
    R05 = 5,
    R06 = 6,

}; // YOU MUST UPDATE the motor_spec_map as well.

extern std::map<uint8_t, MotorSpec> motor_spec_map; // defined in motor.cpp


// constexpr MotorSpec motor_spec_map[4] = {motor_spec_r01, motor_spec_r02, motor_spec_r03, motor_spec_r04};

// extern std::map<uint8_t, std::string> MotorCommunicationTypeToStringMap;

enum MotorModeStatus : uint8_t
{
    Reset = 0,
    Calibration = 1,
    Running = 2,
    Unknown = 255,
};

// 0: Operation control mode 1: Position mode 2: Speed mode 3: Current mode 4: Zero mode uint8
enum MotorRunMode : uint8_t
{
    OperationControlMode = 0,
    PositionMode = 1,
    SpeedMode = 2,
    CurrentMode = 3,
    ZeroMode = 4,
    PositionCSPMode = 5
};

enum MotorParams : uint16_t
{
    // read write
    run_mode = 0X7005,      // 0X7005 0: Operation control mode 1: Position mode 2: Speed mode 3: Current mode 4: Zero mode uint8  1byte
    iq_ref = 0X7006,        // 0X7006 Current mode Iq command               W/R  float   4byte   -23~23A
    mech_vel_ref = 0X700A,  // 0x700A Speed mode speed command              W/R  float   4byte   -30~30rad/s
    torque_limit = 0X700B,  // 0x700B Torque limit                          W/R  float   4byte   {0~12Nm}{0~60Nm}{0~120Nm}
    cur_kp = 0X7010,        // 0x7010 Current Kp                            W/R  float   4byte   Default value 0.125
    cur_ki = 0X7011,        // 0x7011 Current Ki                            W/R  float   4byte   Default value 0.0158
    cur_filt_gain = 0X7014, // 0x7014 Current filter coefficient filt_gain  W/R  float   4byte   0~1.0, default value 0.1
    mech_pos_ref = 0X7016,  // 0x7016 Position mode angle command           W/R  float   4byte   rad
    spd_limit = 0X7017,     // 0x7017 Position mode speed limit             W/R  float   4byte   0~30rad/s
    cur_limit = 0X7018,     // 0x7018 Current mode current limit            W/R  float   4byte   0~23A
    mech_pos = 0X7019,      // 0x7019 Load end count mechanical angle         R  float   4byte   rad
    iqf = 0X701A,           // 0x701A iq filter value                         R  float   4byte   -23~23A
    mech_vel = 0X701B,      // 0x701B Load end count speed                    R  float   4byte   -30~30rad/s
    v_bus = 0X701C,         // 0x701C Bus voltage                             R  float   4byte   V
                            // rotation = 0X701D,      // 0x701D Number of turns                         R  int16   2byte   Number of turns (only for Robstrid01)
    loc_kp = 0X701E,        // 0x701E position kp                           W/R  float   4byte   Default value 30
    spd_kp = 0X701F,        // 0x701F speed kp                              W/R  float   4byte   Default value 1
    spd_ki = 0X7020,        // 0x7020 speed ki                              W/R  float   4byte   Default value 0.002
    spd_filt_gain = 0X7021, // 0x7021 speed filter coefficient filt_gain    W/R  float   4byte   0~1.0, default value 0.1
    zero_sta = 0X7029,      // 0x7029 Zero point flag                       W/R  uint8   1byte   0: 0–2π (default), 1: -π–+π
    CAN_TIMEOUT=0x7028,     // 0X200b CAN timeout 20000 = 1s                W/R  uint32  4byte   0-100000
    };

// extern std::map<uint16_t, std::string> MotorParamsToStringMap;

constexpr MotorParams ALLMOTORPARAMS[] = {
    run_mode,
    iq_ref,
    mech_vel_ref,
    torque_limit,
    cur_kp,
    cur_ki,
    cur_filt_gain,
    mech_pos_ref,
    spd_limit,
    cur_limit,
    mech_pos,
    iqf,
    mech_vel,
    v_bus,
    loc_kp,
    spd_kp,
    spd_ki,
    spd_filt_gain,
    CAN_TIMEOUT};


inline void timespec_add_ns(timespec& ts, long ns) {
    ts.tv_nsec += ns;
    if (ts.tv_nsec >= 1'000'000'000L) {
        ts.tv_sec += 1;
        ts.tv_nsec -= 1'000'000'000L;
    }
}

inline void precise_periodic_sleep(timespec& next, long interval_ns) {
    timespec_add_ns(next, interval_ns);
    clock_nanosleep(CLOCK_MONOTONIC, TIMER_ABSTIME, &next, nullptr);
}


inline std::tm get_time_now()
{
    auto now = std::chrono::system_clock::to_time_t(std::chrono::system_clock::now());
    return *std::localtime(&now); // Dereference here to return the tm struct by value
}

inline fmt::text_style fmt_tyle_ok()
{
    return fmt::fg(fmt::color::green);
    // return fmt::emphasis::bold | fmt::fg(fmt::color::green);
}

inline fmt::text_style fmt_tyle_warn()
{
    return fmt::fg(fmt::color::yellow);
    // return fmt::emphasis::bold | fmt::fg(fmt::color::yellow);
}

inline fmt::text_style fmt_style_error()
{
    return fmt::emphasis::bold | fmt::fg(fmt::color::red);
}


class FpsCounter {
    public:
        /**
         * @brief Constructs an FpsCounter object.
         *
         * @param print_interval_ms The interval in milliseconds at which to calculate and print FPS.
         * Defaults to 1000 ms (1 second).
         * @param label An optional string label to prepend to the FPS output.
         */
        explicit FpsCounter(long long print_interval_ms = 1000, const std::string& label = "")
            : print_interval_ms_(print_interval_ms), label_(label), frame_count_(0) {
            // Initialize the start time when the object is created
            start_time_ = std::chrono::high_resolution_clock::now();
        }
    
        /**
         * @brief Call this method once per frame or iteration of your loop.
         *
         * This method increments the frame count and, if the print interval
         * has passed, calculates and prints the current FPS before resetting
         * the counters for the next interval.
         */
        void tick() {
            // Increment the frame count for this tick (frame)
            frame_count_++;
    
            // Get the current time
            auto current_time = std::chrono::high_resolution_clock::now();
            // Calculate the elapsed time in milliseconds since the last reset
            auto elapsed_ms = std::chrono::duration_cast<std::chrono::milliseconds>(current_time - start_time_).count();
    
            // Check if the print interval has passed
            if (elapsed_ms >= print_interval_ms_) {
                // Calculate FPS: frames / (elapsed_time_in_seconds)
                // Ensure floating-point division by casting
                double fps = static_cast<double>(frame_count_) / (elapsed_ms / 1000.0);
    
                // Print the calculated FPS with the optional label
                if (!label_.empty()) {
                    fmt::print("{}: {:.2f} FPS\n", label_, fps);
                } else {
                    fmt::print("{:.2f} FPS\n", fps);
                }
    
                // Reset the frame count and start time for the next measurement interval
                frame_count_ = 0;
                start_time_ = current_time;
            }
        }
    
    private:
        std::chrono::high_resolution_clock::time_point start_time_; // Time when the current interval started
        long long frame_count_;                                  // Number of frames in the current interval
        long long print_interval_ms_;                            // How often to print FPS (in milliseconds)
        std::string label_;                                      // Optional label for the output
    };

bool setup_socket(const std::string &interface, int &socket_fd);
void send_extended_frame(int socket_fd, uint32_t can_id, const uint8_t *data, uint8_t len);
void receive_frame(int socket_fd);
// void receive_frame_thread(int socket_fd);
void print_frame(can_frame &frame, std::string header = "");

struct CanMessage : can_frame
{
    CanMessage();

    void msgGetDeviceMcuId(const uint8_t &master_can_id, const uint8_t &target_can_id);
    void msgChangeCanId(const uint8_t &master_can_id, const uint8_t &target_can_id, const uint8_t &new_target_can_id);
    void msgEnableMotor(const uint8_t &master_can_id, const uint8_t &target_can_id);
    void msgDisableMotor(const uint8_t &master_can_id, const uint8_t &target_can_id, bool clear_fault = false);
    void msgMotionControl(const uint8_t &target_can_id, const uint16_t &torque, const uint16_t &pos, const uint16_t &vel, const uint16_t &kp, const uint16_t &kd);
    void msgSetPosZero(const uint8_t &master_can_id, const uint8_t &target_can_id);
    void msgReadParam(const uint8_t &master_can_id, const uint8_t &target_can_id, const uint16_t &motor_param);
    void msgSaveData(const uint8_t &master_can_id, const uint8_t &target_can_id);

    template<typename T>
    void msgWriteParam(const uint8_t &master_can_id, const uint8_t &target_can_id, const uint16_t &motor_param,T ref){
        this->can_id = MotorCommunicationType::SetSingleParameter << 24 | master_can_id << 8 | target_can_id;
        memset(data, 0, sizeof(data));
        memcpy(&data[0],&motor_param,sizeof(motor_param));
        memcpy(&data[4],&ref,sizeof(ref));
        // // --- Using typeid ---
        // fmt::print("DEBUG: sizeof(motor_param)={0}, sizeof(ref)={1}, type(T)='{2}'\n",
        //     sizeof(motor_param),
        //     sizeof(ref),
        //     typeid(T).name()); // Get the name using typeid
     
    }
    
    bool send(const int &socket_fd, bool should_print = true);  // returns true on success
};

class CanController
{
public:
    int socket_fd;
    std::string can_interface = "";

    // Thread-safe queue for CAN frames
    std::queue<can_frame> queue_recv;
    std::mutex mutex_recv;
    uint64_t counter_sent = 0;
    uint64_t counter_received = 0;
    bool receive_thread_should_run = true;
    bool started = false;
    std::thread receiver_thread;

    CanController()
    {
        fmt::print(fmt_tyle_ok(), "[{0:%F %T}][{1}:{2}][{3}]: constructor called.\033[0m\033[0m\n", get_time_now(), __FILE__, __LINE__, can_interface.c_str());
    }

    // CanController(std::string can_interface);
    ~CanController();

    void start();

    void send(can_frame &frame);                                                 // send 1 frame
    can_frame receive(int timeout_millisecond = 200, int wait_microsecond = 10); // receive 1 frame
private:
    void receive_frame_thread();
};

/*******************************************************************************/
float uint16_to_float(uint16_t x, float x_min, float x_max, int bits);
int float_to_uint(float x, float x_min, float x_max, int bits);
uint16_t float_to_uint16(float x, const float &x_min, const float &x_max);
uint16_t float_to_uint16_loop(float x, const float &x_min, const float &x_max);

float byte_to_float(uint8_t *bytedata);
uint16_t byte_to_uint16(uint8_t *bytedata);
uint32_t byte_to_uint32(uint8_t *bytedata);

class CanMotorController //: public CanController
{
public:
    int num_motors;
    uint8_t master_can_id = 0xFD;

    std::vector<std::string> can_interfaces;
    Eigen::Matrix<uint8_t, Eigen::Dynamic, 1> can_interface_ids; // the index of the can interface of the motor
    Eigen::Matrix<uint8_t, Eigen::Dynamic, 1> can_ids;           // the can id of the motor
    Eigen::Matrix<uint8_t, Eigen::Dynamic, 1> motor_types;

    std::vector<MotorSpec> motor_specs;
    std::array<int8_t, 256> can_id_to_idx; // O(1) CAN ID → motor index (-1 = unknown)

    Eigen::Matrix<uint8_t, Eigen::Dynamic, 1> run_mode;    // (wr) 0X7005 0: Operation control mode 1: Position mode 2: Speed mode 3: Current mode 4: Zero mode uint8  1byte
    Eigen::Matrix<float, Eigen::Dynamic, 1> iq_ref;        // (wr) 0X7006 Current mode Iq command               W/R  float   4byte   -23~23A
    Eigen::Matrix<float, Eigen::Dynamic, 1> mech_vel_ref;  // (wr) 0x700A Speed mode speed command              W/R  float   4byte   -30~30rad/s
    Eigen::Matrix<float, Eigen::Dynamic, 1> torque_limit;  // (wr) 0x700B Torque limit                          W/R  float   4byte   {0~12Nm}{0~60Nm}{0~120Nm}
    Eigen::Matrix<float, Eigen::Dynamic, 1> cur_kp;        // (wr) 0x7010 Current Kp                            W/R  float   4byte   Default value 0.125
    Eigen::Matrix<float, Eigen::Dynamic, 1> cur_ki;        // (wr) 0x7011 Current Ki                            W/R  float   4byte   Default value 0.0158
    Eigen::Matrix<float, Eigen::Dynamic, 1> cur_filt_gain; // (wr) 0x7014 Current filter coefficient filt_gain  W/R  float   4byte   0~1.0, default value 0.1
    Eigen::Matrix<float, Eigen::Dynamic, 1> mech_pos_ref;  // (wr) 0x7016 Position mode angle command           W/R  float   4byte   rad
    Eigen::Matrix<float, Eigen::Dynamic, 1> spd_limit;     // (wr) 0x7017 Position mode speed limit             W/R  float   4byte   0~30rad/s
    Eigen::Matrix<float, Eigen::Dynamic, 1> cur_limit;     // (wr) 0x7018 Current mode current limit            W/R  float   4byte   0~23A
    Eigen::Matrix<float, Eigen::Dynamic, 1> mech_pos;      // (wr) 0x7019 Load end count mechanical angle         R  float   4byte   rad
    Eigen::Matrix<float, Eigen::Dynamic, 1> iqf;           // (ro)  0x701A iq filter value                         R  float   4byte   -23~23A
    Eigen::Matrix<float, Eigen::Dynamic, 1> mech_vel;      // (ro)  0x701B Load end count speed                    R  float   4byte   -30~30rad/s
    Eigen::Matrix<float, Eigen::Dynamic, 1> v_bus;         // (ro)  0x701C Bus voltage                             R  float   4byte   V
    // Eigen::Matrix<float, Eigen::Dynamic, 1> rotation;    // (ro)  0x701D Number of turns                         R  int16   2byte   Number of turns (only for Robstrid01)
    Eigen::Matrix<float, Eigen::Dynamic, 1> loc_kp;        // (wr) 0x701E position kp                           W/R  float   4byte   Default value 30
    Eigen::Matrix<float, Eigen::Dynamic, 1> spd_kp;        // (wr) 0x701F speed kp                              W/R  float   4byte   Default value 1
    Eigen::Matrix<float, Eigen::Dynamic, 1> spd_ki;        // (wr) 0x7020 speed ki                              W/R  float   4byte   Default value 0.002
    Eigen::Matrix<float, Eigen::Dynamic, 1> spd_filt_gain; // (wr) 0x7021 speed filter coefficient filt_gain    W/R  float   4byte   0~1.0, default value 0.1
    Eigen::Matrix<uint32_t, Eigen::Dynamic, 1> can_timeout; // (wr) 0x7028 CAN timeout 20000 = 1s              W/R  uint32  4byte   0-100000

    Eigen::Matrix<float, Eigen::Dynamic, 1> mech_torque_ref; // (wr) feedforward load end torque (desired torque)
    Eigen::Matrix<float, Eigen::Dynamic, 1> mech_torque;     // (ro) load end torque
    Eigen::Matrix<float, Eigen::Dynamic, 1> temperature;     // (ro) temperature [degC]

    Eigen::Matrix<uint8_t, Eigen::Dynamic, 1> mode_status; // // 0: reset, 1: calibration, 2: running

    Eigen::Matrix<int, Eigen::Dynamic, 1> socket_fds;

    // Thread-safe queue for CAN frames
    std::queue<can_frame> queue_recv;
    std::mutex mutex_recv;
    std::atomic<bool> receive_thread_should_run{true};
    std::vector<std::thread> receiver_threads;
    std::vector<std::thread> sender_threads;
    std::thread motion_control_thread;
    bool motion_control_thread_started = false;

    // whether to print debug messages
    bool should_print_send = true;
    bool should_print_recv = true;

    // Latency stats
    LatencyLogger latency_logger;

    int max_try_send = 50; // maximum number of tries to send the same CAN frame
    int response_wait_microsecond = 4000; // typical waiting time needed for motor to response
    int min_same_bus_delay_us = 40; // minimum delay between sends on same CAN bus (busy wait)
    // Receiver poll timeout. Start long (2000ms) to tolerate startup quiet gaps before motors
    // are enabled. After start_motion_control_continuously(), tighten to ~200ms for runtime
    // safety — at 100Hz the control loop generates frames every ~10ms so this never fires.
    int recv_timeout_ms = 2000; // plain int is fine: 32-bit reads/writes are hardware-atomic on ARM64, and poll() prevents compiler hoisting

    // Delete copy constructor and assignment operator
    CanMotorController(const CanMotorController &) = delete;
    CanMotorController &operator=(const CanMotorController &) = delete;

    // Allow move constructor and assignment operator
    CanMotorController(CanMotorController &&) noexcept = default;
    CanMotorController &operator=(CanMotorController &&) noexcept = default;

    CanMotorController(
        std::vector<std::string> can_interfaces,
        Eigen::Matrix<uint8_t, Eigen::Dynamic, 1> can_interface_ids,
        Eigen::Matrix<uint8_t, Eigen::Dynamic, 1> can_ids,
        Eigen::Matrix<uint8_t, Eigen::Dynamic, 1> motor_types)
        // : can_controllers(can_interfaces.size())
        : socket_fds(can_ids.size()),
          receiver_threads(can_interfaces.size()),
          sender_threads(can_interfaces.size()),
          run_mode(can_ids.size()),
          iq_ref(can_ids.size()),
          mech_vel_ref(can_ids.size()),
          torque_limit(can_ids.size()),
          cur_kp(can_ids.size()),
          cur_ki(can_ids.size()),
          cur_filt_gain(can_ids.size()),
          mech_pos_ref(can_ids.size()),
          spd_limit(can_ids.size()),
          cur_limit(can_ids.size()),
          mech_pos(can_ids.size()),
          iqf(can_ids.size()),
          mech_vel(can_ids.size()),
          v_bus(can_ids.size()),
          loc_kp(can_ids.size()),
          spd_kp(can_ids.size()),
          spd_ki(can_ids.size()),
          spd_filt_gain(can_ids.size()),
          can_timeout(can_ids.size()),
          mech_torque(can_ids.size()),
          mech_torque_ref(can_ids.size()),
          temperature(can_ids.size()),
          mode_status(can_ids.size()),
          latency_logger(can_ids.size(), 600)
    {

        this->can_interfaces = can_interfaces;
        this->can_interface_ids = can_interface_ids;
        this->can_ids = can_ids;
        this->num_motors = can_ids.size();
        this->motor_types = motor_types;

        // initialize vectors with zeros
        mech_torque_ref.setZero();
        mech_vel_ref.setZero();
        mech_pos_ref.setZero();
        mode_status.setZero();

        // Build lookup tables BEFORE spawning threads — parseFrame() reads these
        can_id_to_idx.fill(-1);
        motor_specs.resize(num_motors);
        for (int i = 0; i < num_motors; i++) {
            can_id_to_idx[can_ids(i)] = (int8_t)i;
            motor_specs[i] = motor_spec_map[motor_types(i)];
        }

        build_round_robin_order();

        // Lock memory to prevent page faults (requires root/CAP_SYS_NICE)
        if (mlockall(MCL_CURRENT | MCL_FUTURE) != 0) {
            fmt::print(fmt_tyle_warn(), "[{0:%F %T}][{1}:{2}][{3}]: Failed to lock memory: {4}\033[0m\n",
                       get_time_now(), __FILE__, __LINE__, __func__, strerror(errno));
            fmt::print(fmt_tyle_warn(), "  → Running without RT memory locking (may have page fault latency spikes)\n");
            fmt::print(fmt_tyle_warn(), "  → To enable: run with sudo OR setcap cap_ipc_lock=+ep <program>\033[0m\n");
        }

        // Start CAN sockets and receiver threads
        for (size_t i = 0; i < can_interfaces.size(); ++i)
        {
            if (!setup_socket(can_interfaces[i], socket_fds(i))) {
                fmt::print(fmt_style_error(), "[{0:%F %T}][{1}:{2}][{3}]: Failed to setup socket for {4}\033[0m\n",
                           get_time_now(), __FILE__, __LINE__, __func__, can_interfaces[i]);
                throw std::runtime_error("Socket setup failed for " + can_interfaces[i]);
            }
            receiver_threads[i] = std::thread(&CanMotorController::receive_frame_thread, this, i);
        }

        // Initialize per-bus send contexts, staging buffers, and spawn sender threads
        bus_send_contexts.reserve(can_interfaces.size());
        staged_frames.resize(can_interfaces.size());
        for (size_t i = 0; i < can_interfaces.size(); ++i) {
            bus_send_contexts.push_back(std::make_unique<BusSendContext>());
            sender_threads[i] = std::thread(&CanMotorController::send_frame_thread, this, (int)i);
        }

    }

    ~CanMotorController()
    {
        receive_thread_should_run = false;

        // Join motion_control_thread FIRST (before closing sockets)
        // to prevent "Bad file descriptor" errors from sending on closed sockets
        if (motion_control_thread_started){
            if (motion_control_thread.joinable())
            {
                motion_control_thread.join();
                fmt::print(fmt_tyle_ok(), "[{0:%F %T}][{1}:{2}][{3}]: joining motion_control_thread success.\n", get_time_now(), __FILE__, __LINE__, __func__);
            }
            else
            {
                fmt::print(fmt_tyle_warn(), "[{0:%F %T}][{1}:{2}][{3}]: motion_control_thread not joinable.\033[0m\n", get_time_now(), __FILE__, __LINE__, __func__);
            }
        }

        // drain sender threads before closing sockets (so no send on closed fd)
        for (auto& ctx : bus_send_contexts)
            ctx->cv.notify_all();
        for (size_t i = 0; i < can_interfaces.size(); ++i)
        {
            if (sender_threads[i].joinable())
            {
                sender_threads[i].join();
                fmt::print(fmt_tyle_ok(), "[{0:%F %T}][{1}:{2}][{3}]: joined sender thread #{4}.\033[0m\n", get_time_now(), __FILE__, __LINE__, __func__, i);
            }
        }

        // shut down receiver threads and sockets
        for (size_t i = 0; i < can_interfaces.size(); ++i)
        {
            // joining thread
            if (receiver_threads[i].joinable())
            {
                receiver_threads[i].join();
                fmt::print(fmt_tyle_ok(), "[{0:%F %T}][{1}:{2}][{3}]: shutdown receiver thread #{4}.\033[0m\n", get_time_now(), __FILE__, __LINE__, __func__, i);
            }
            else
            {
                fmt::print(fmt_tyle_warn(), "[{0:%F %T}][{1}:{2}][{3}]: receiver thread #{4} not joinable.\033[0m\n", get_time_now(), __FILE__, __LINE__, __func__, i);
            }
            // closing socket
            int flags = fcntl(socket_fds(i), F_GETFL, 0);
            if (flags != -1)
            {
                close(socket_fds(i));
                fmt::print(fmt_tyle_ok(), "[{0:%F %T}][{1}:{2}][{3}]: close socket #{4}.\033[0m\n", get_time_now(), __FILE__, __LINE__, __func__, i);
            }
            else
            {
                fmt::print(fmt_tyle_warn(), "[{0:%F %T}][{1}:{2}][{3}]: cannot close socket #{4}.\033[0m\n", get_time_now(), __FILE__, __LINE__, __func__, i);
            }
        }

        fmt::print(fmt_tyle_ok(), "[{0:%F %T}][{1}:{2}][{3}]: shutdown success.\033[0m\n", get_time_now(), __FILE__, __LINE__, __func__);
    }


    void getAllParams()
    {
        // initialize eigne matrix with num_motors
        // Eigen::Matrix<uint8_t, Eigen::Dynamic, 1> num_msgs_sent(num_motors);
        // num_msgs_sent.setZero();

        for (int id : round_robin_motor_order)
        {
            CanMessage msg;
            msg.msgGetDeviceMcuId(master_can_id, can_ids[id]);
            send(id, msg);
        }
        std::this_thread::sleep_for(std::chrono::microseconds(response_wait_microsecond));

        for (auto &param : ALLMOTORPARAMS)
        {
            for (int id : round_robin_motor_order)
            {
                CanMessage msg;
                msg.msgReadParam(master_can_id, can_ids[id], param);
                std::this_thread::sleep_for(std::chrono::microseconds(10));
                send(id, msg);
            }
            std::this_thread::sleep_for(std::chrono::microseconds(response_wait_microsecond));
        }
    }

    // void receive()
    // {
    //     // timer, start time
    //     auto start_time = std::chrono::high_resolution_clock::now();
    //     can_frame frame;
    //     while (receive_thread_should_run)
    //     {
    //         // fmt::print("receive_thread_should_run={0}\033[0m\n",receive_thread_should_run);
    //         if (!queue_recv.empty())
    //         {
    //             {
    //                 std::lock_guard<std::mutex> lock(mutex_recv);
    //                 frame = std::move(queue_recv.front());
    //                 queue_recv.pop();
    //             }
    //             start_time = std::chrono::high_resolution_clock::now();
    //             fmt::print("[RECV][ID] 0x{:x} [Data]", frame.can_id);
    //             for (int i = 0; i < frame.len; i++)
    //             {
    //                 fmt::print(" {:02x}", frame.data[i]);
    //             }
    //             fmt::print("\n");
    //             parseFrame(frame);
    //         }
    //         auto now = std::chrono::high_resolution_clock::now();
    //         auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(now - start_time);
    //         // fmt::print("elapsed: {} ms\n", elapsed.count());
    //         if (elapsed.count() > 500)
    //         {
    //             fmt::print(fmt_tyle_warn(), "[{0:%F %T}][{1}:{2}][{3}]: Timeout waiting for CAN frame.\033[0m\n", get_time_now(), __FILE__, __LINE__, __func__);
    //             receive_thread_should_run = false;
    //             break;
    //         }
    //         // print exit
    //         // ParseMotorData(frame, motor_spec_map);
    //     }
    // }

    void getParam(const uint16_t param)
    {
        for (int i = 0; i < num_motors; i++)
        {
        CanMessage msg;
        msg.msgReadParam(master_can_id, can_ids[i], param);
        send(i, msg);
        }
        std::this_thread::sleep_for(std::chrono::microseconds(response_wait_microsecond));
        
        // fmt::print(fmt_tyle_ok(), "[{0:%F %T}][{1}:{2}][{3}]: exit.\033[0m\n", get_time_now(), __FILE__, __LINE__, __func__);
    }

    template<typename T>
    void setParam(const uint16_t param, Eigen::Matrix<T, Eigen::Dynamic, 1> ref){
        for (int i = 0; i < num_motors; i++)
        {
        CanMessage msg;
        msg.msgWriteParam<T>(master_can_id, can_ids[i], param, ref(i));
        send(i, msg);
        }
        std::this_thread::sleep_for(std::chrono::microseconds(response_wait_microsecond));
    }

    // method to get the stats
    std::vector<float> get_motor_latency_history(int motor_id) {
        return latency_logger.get_history(motor_id);
    }

    // Number of frames queued but not yet sent per bus — non-zero indicates a send backlog
    int get_bus_send_queue_depth(int bus_id) {
        auto& ctx = *bus_send_contexts[bus_id];
        std::lock_guard<std::mutex> lock(ctx.mutex);
        return (int)ctx.queue.size();
    }

    void parseFrame(can_frame &frame)
    {
        uint8_t communication_type = (frame.can_id & 0x1F000000) >> 24;
        uint8_t target_can_id = (frame.can_id & 0xFFFF00) >> 8;
        int8_t id = can_id_to_idx[target_can_id]; // O(1) array lookup
        if (id < 0) {
            fmt::print(stderr, fmt_style_error(), "[{0:%F %T}][{1}:{2}][{3}]: Unknown CAN ID: 0x{4:x} (frame.can_id=0x{5:x})\033[0m\n",
                       get_time_now(), __FILE__, __LINE__, __func__, target_can_id, frame.can_id);
            return;
        }

        if(communication_type == MotorCommunicationType::MotorStatus){
            // Calculate latency (using seconds)
            double now = std::chrono::duration<double>(std::chrono::steady_clock::now().time_since_epoch()).count();
            latency_logger.log_end(id, now);
        }

        switch (communication_type)
        {
        case MotorCommunicationType::MotorStatus:
        {
            // uint8_t master_can_id = (frame.can_id & 0xFF);
            const auto &spec = motor_specs[id];
            mech_pos(id) = uint16_to_float(frame.data[0] << 8 | frame.data[1], spec.pos_min, spec.pos_max, 16);
            mech_vel(id) = uint16_to_float(frame.data[2] << 8 | frame.data[3], spec.vel_min, spec.vel_max, 16);
            mech_torque(id) = uint16_to_float(frame.data[4] << 8 | frame.data[5], spec.torque_min, spec.torque_max, 16);
            temperature(id) = (float)(frame.data[6] << 8 | frame.data[7]) * 0.1f;
            uint8_t error_code = uint8_t((frame.can_id & 0x3F0000) >> 16);
            mode_status(id) = uint8_t((frame.can_id & 0xC00000) >> 22); // 0: reset, 1: calibration, 2: running

            if (error_code != 0)
            {
                fmt::print(fmt_style_error(), "[{0:%F %T}][{1}:{2}][{3}]: motor#{4:02d} Error:{5}{6}{7}{8}{9}{10}\033[0m\n",
                           get_time_now(), __FILE__, __LINE__, __func__, id,
                           (error_code & 0b1) ? "[UNDER VOLTAGE]" : "",
                           (error_code & 0b10) ? "[OVER CURRENT]" : "",
                           (error_code & 0b100) ? "[OVER TEMPERATURE]" : "",
                           (error_code & 0b1000) ? "[MEGNETIC ENCODING FAULT]" : "",
                           (error_code & 0b10000) ? "[HALL ENCODING FAULT]" : "",
                           (error_code & 0b100000) ? "[UNCALIBRATED]" : "");
            }
            if (should_print_recv)
                fmt::print("[RECV] motor#{:02d}: can_id: 0x{:x} pos: {:.3f} rad, vel: {:.2f} rad/s, torque: {:.1f} Nm, temp: {:.1f} °C, error_code: {}, pattern: {}\n",
                           id,target_can_id, mech_pos(id), mech_vel(id), mech_torque(id), temperature(id), error_code, mode_status(id));
            break;
        }
        case MotorCommunicationType::GetCanID:
        {
            // mcu ids
            uint8_t least_id_byte = (frame.can_id & 0xFF);
            assert(least_id_byte == 0xFE);
            uint64_t mcu_id;
            std::memcpy(&mcu_id, frame.data, sizeof(mcu_id));
            if (should_print_recv)
                fmt::print("[RECV] motor#{:02d} mcu_id: 0x{:016x}\n", id, mcu_id);
            break;
        }
        case MotorCommunicationType::GetSingleParameter:
        {
            uint16_t motor_param = frame.data[1] << 8 | frame.data[0];
            switch (motor_param)
            {
            case MotorParams::CAN_TIMEOUT:{
                can_timeout(id) = byte_to_uint32(frame.data);
                if (should_print_recv)
                    fmt::print("[RECV] motor#{:02d} param: 0x{:04x}: CAN_TIMEOUT: {}\n", id, motor_param, can_timeout(id));
                break;
            }
            case 0X301c:{ // TODO: CHECK IF THIS IS CORRECT
                uint32_t timeout = byte_to_uint32(frame.data);
                fmt::print("[RECV] motor#{:02d} param: 0x{:04x}: CAN_TIMEOUT timer: {}\n", id, motor_param, timeout);
                break;
            }
            case MotorParams::run_mode:
            {
                run_mode(id) = uint8_t(frame.data[4]);
                if (should_print_recv)
                    fmt::print("[RECV] motor#{:02d} param: 0x{:04x}: run mode: {}\n", id, motor_param, run_mode(id));
                break;
            }
            case MotorParams::iq_ref:
            {
                iq_ref(id) = byte_to_float(frame.data);
                if (should_print_recv)
                    fmt::print("[RECV] motor#{:02d} param: 0x{:04x}: iq ref: {:.2f}\n", id, motor_param, iq_ref(id));
                break;
            }
            case MotorParams::mech_vel_ref:
            {
                mech_vel_ref(id) = byte_to_float(frame.data);
                if (should_print_recv)
                    fmt::print("[RECV] motor#{:02d} param: 0x{:04x}: spd ref: {:.2f}\n", id, motor_param, mech_vel_ref(id));
                break;
            }
            case MotorParams::torque_limit:
            {
                torque_limit(id) = byte_to_float(frame.data);
                if (should_print_recv)
                    fmt::print("[RECV] motor#{:02d} param: 0x{:04x}: torque limit: {:.2f}\n", id, motor_param, torque_limit(id));
                break;
            }
            case MotorParams::cur_kp:
            {
                cur_kp(id) = byte_to_float(frame.data);
                if (should_print_recv)
                    fmt::print("[RECV] motor#{:02d} param: 0x{:04x}: cur kp: {:.2f}\n", id, motor_param, cur_kp(id));
                break;
            }
            case MotorParams::cur_ki:
            {
                cur_ki(id) = byte_to_float(frame.data);
                if (should_print_recv)
                    fmt::print("[RECV] motor#{:02d} param: 0x{:04x}: cur ki: {:.2f}\n", id, motor_param, cur_ki(id));
                break;
            }
            case MotorParams::cur_filt_gain:
            {
                cur_filt_gain(id) = byte_to_float(frame.data);
                if (should_print_recv)
                    fmt::print("[RECV] motor#{:02d} param: 0x{:04x}: cur filt gain: {:.2f}\n", id, motor_param, cur_filt_gain(id));
                break;
            }
            case MotorParams::mech_pos_ref:
            {
                mech_pos_ref(id) = byte_to_float(frame.data);
                if (should_print_recv)
                    fmt::print("[RECV] motor#{:02d} param: 0x{:04x}: loc ref: {:.2f}\n", id, motor_param, mech_pos_ref(id));
                break;
            }
            case MotorParams::spd_limit:
            {
                spd_limit(id) = byte_to_float(frame.data);
                if (should_print_recv)
                    fmt::print("[RECV] motor#{:02d} param: 0x{:04x}: limit spd: {:.2f}\n", id, motor_param, spd_limit(id));
                break;
            }
            case MotorParams::cur_limit:
            {
                cur_limit(id) = byte_to_float(frame.data);
                if (should_print_recv)
                    fmt::print("[RECV] motor#{:02d} param: 0x{:04x}: cur_limit: {:.2f}\n", id, motor_param, cur_limit(id));
                break;
            }
            case MotorParams::mech_pos:
            {
                mech_pos(id) = byte_to_float(frame.data);
                if (should_print_recv)
                    fmt::print("[RECV] motor#{:02d} param: 0x{:04x}: mech_pos: {:.2f}\n", id, motor_param, mech_pos(id));
                break;
            }
            case MotorParams::iqf:
            {
                iqf(id) = byte_to_float(frame.data);
                if (should_print_recv)
                    fmt::print("[RECV] motor#{:02d} param: 0x{:04x}: iqf: {:.2f}\n", id, motor_param, iqf(id));
                break;
            }
            case MotorParams::mech_vel:
            {
                mech_vel(id) = byte_to_float(frame.data);
                if (should_print_recv)
                    fmt::print("[RECV] motor#{:02d} param: 0x{:04x}: mech_vel: {:.2f}\n", id, motor_param, mech_vel(id));
                break;
            }
            case MotorParams::v_bus:
            {
                v_bus(id) = byte_to_float(frame.data);
                if (should_print_recv)
                    fmt::print("[RECV] motor#{:02d} param: 0x{:04x}: v_bus: {:.2f}\n", id, motor_param, v_bus(id));
                break;
            }
            // case MotorParams::rotation:
            // {
            //     rotation(id) = frame.data[1] << 8 | frame.data[0];
            //     if(should_print_recv) printf("rotation: %d\n", rotation(id));
            //     break;
            // }
            case MotorParams::loc_kp:
            {
                loc_kp(id) = byte_to_float(frame.data);
                if (should_print_recv)
                    fmt::print("[RECV] motor#{:02d} param: 0x{:04x}: loc kp: {:.2f}\n", id, motor_param, loc_kp(id));
                break;
            }
            case MotorParams::spd_kp:
            {
                spd_kp(id) = byte_to_float(frame.data);
                if (should_print_recv)
                    fmt::print("[RECV] motor#{:02d} param: 0x{:04x}: spd kp: {:.2f}\n", id, motor_param, spd_kp(id));
                break;
            }
            case MotorParams::spd_ki:
            {
                spd_ki(id) = byte_to_float(frame.data);
                if (should_print_recv)
                    fmt::print("[RECV] motor#{:02d} param: 0x{:04x}: spd ki: {:.2f}\n", id, motor_param, spd_ki(id));
                break;
            }
            case MotorParams::spd_filt_gain:
            {
                spd_filt_gain(id) = byte_to_float(frame.data);
                if (should_print_recv)
                    fmt::print("[RECV] motor#{:02d} param: 0x{:04x}: spd filt gain: {:.2f}\n", id, motor_param, spd_filt_gain(id));
                break;
            }
            case MotorParams::zero_sta:
            {
                uint8_t zero_sta_val = uint8_t(frame.data[4]);
                if (should_print_recv)
                    fmt::print("[RECV] motor#{:02d} param: 0x{:04x}: zero_sta: {} (0=0-2π, 1=-π-+π)\n", id, motor_param, zero_sta_val);
                break;
            }
            default:
                break;
            }
            break;
        }
        case MotorCommunicationType::Error: // error frame
        {
            fmt::print(fmt_style_error(), "[RECV] error frame: 0x{:04x}\033[0m\n", frame.can_id);
            break;
        }
        default:
        {
            fmt::print(fmt_style_error(), "[{0:%F %T}][{1}:{2}][{3}]: received unknown frame: 0x{4:04x}.\033[0m\n", get_time_now(), __FILE__, __LINE__, __func__, frame.can_id);
        }
        };
    }

    void start_motion_control_continuously()
    {
        motion_control_thread = std::thread([this]() {
            // Set RT priority for motion control thread (requires root/CAP_SYS_NICE)
            struct sched_param param;
            param.sched_priority = 90;
            if (sched_setscheduler(0, SCHED_FIFO, &param) != 0) {
                fmt::print(fmt_tyle_warn(), "[{0:%F %T}][{1}:{2}][{3}]: Failed to set RT priority for motion control: {4}\033[0m\n",
                           get_time_now(), __FILE__, __LINE__, __func__, strerror(errno));
                fmt::print(fmt_tyle_warn(), "  → Running without RT scheduling (no guaranteed timing)\n");
                fmt::print(fmt_tyle_warn(), "  → To enable: run with sudo OR setcap cap_sys_nice=+ep <program>\033[0m\n");
            }
            // Run the control loop
            motion_control_continuously();
        });
        motion_control_thread_started = true;
    }

    void stop()
    {
        receive_thread_should_run = false;
        fmt::print(fmt_tyle_ok(), "[{0:%F %T}][{1}:{2}][{3}]: stop() called, signaling threads to exit.\033[0m\n", get_time_now(), __FILE__, __LINE__, __func__);
    }

    void motion_control_continuously() {
        // FpsCounter fps_counter(500, "[motion_control_continuously] control Loop"); // Print every 500ms with a label
        while (receive_thread_should_run){
            // compute fps and print
            motion_control_once();

            // fps_counter.tick();

            std::this_thread::sleep_for(std::chrono::microseconds(response_wait_microsecond));
        }
        receive_thread_should_run = false;
        fmt::print(fmt_tyle_ok(), "[{0:%F %T}][{1}:{2}]: motion_control_thread exit success.\033[0m\n", get_time_now(), __FILE__, __LINE__, __func__);
    }

    void motion_control_once() {
        double now = std::chrono::duration<double>(std::chrono::steady_clock::now().time_since_epoch()).count();

        // Phase 1: compute all frames and stage per bus — no mutex held
        for (auto& buf : staged_frames) buf.clear();
        for (int id : round_robin_motor_order) {
            const auto &spec = motor_specs[id];
            // NaN guard: float_to_uint16 clamps numeric out-of-range but NaN bypasses both
            // comparisons → UB in the cast. Warn loudly; fallbacks are semantically safe.
            auto chk = [&](float v, float fb, const char* nm) {
                if (__builtin_expect(std::isnan(v), 0)) {
                    fmt::print(fmt_style_error(), "[SAFETY] motor#{:02d} NaN {} → {:.3f}\n", id, nm, fb);
                    return fb;
                }
                return v;
            };
            u_int16_t mech_torque_u16 = float_to_uint16(chk(mech_torque_ref(id), 0.0f,         "torq"), spec.torque_min, spec.torque_max);
            u_int16_t mech_pos_u16    = float_to_uint16(chk(mech_pos_ref(id),    mech_pos(id), "pos"),  spec.pos_min,    spec.pos_max);
            u_int16_t mech_vel_u16    = float_to_uint16(chk(mech_vel_ref(id),    0.0f,         "vel"),  spec.vel_min,    spec.vel_max);
            u_int16_t kp_u16          = float_to_uint16(chk(loc_kp(id),          0.0f,         "kp"),   spec.kp_min,     spec.kp_max);
            u_int16_t kd_u16          = float_to_uint16(chk(spd_kp(id),          0.0f,         "kd"),   spec.kd_min,     spec.kd_max);

            CanMessage frame;
            frame.msgMotionControl(can_ids(id), mech_torque_u16, mech_pos_u16, mech_vel_u16, kp_u16, kd_u16);

            latency_logger.log_start(id, now);
            staged_frames[can_interface_ids(id)].push_back(std::move(frame));
        }

        // Phase 2: bulk-enqueue per bus with one lock + one notify each
        // → sender threads for all buses start simultaneously
        for (size_t bus = 0; bus < can_interfaces.size(); ++bus) {
            if (staged_frames[bus].empty()) continue;
            {
                std::lock_guard<std::mutex> lock(bus_send_contexts[bus]->mutex);
                for (auto& f : staged_frames[bus])
                    bus_send_contexts[bus]->queue.push(std::move(f));
            }
            bus_send_contexts[bus]->cv.notify_one();
        }
    }


    // Enable motors where motor_enable_mask[id] is true; unmasked motors are left in their current state.
    void enable(const Eigen::Matrix<bool, Eigen::Dynamic, 1>& motor_enable_mask)
    {
        assert(motor_enable_mask.size() == num_motors);
        MotorModeStatus target_status = MotorModeStatus::Running;
        for (int id = 0; id < num_motors; id++)
            if (motor_enable_mask(id)) mode_status(id) = MotorModeStatus::Unknown;
        bool success;
        int try_count = -1;
        while (++try_count <= max_try_send)
        {
            success = true;
            for (int id = 0; id < num_motors; id++)
            {
                if (!motor_enable_mask(id)) continue;
                if (mode_status(id) != target_status)
                {
                    CanMessage frame;
                    frame.msgEnableMotor(master_can_id, can_ids(id));
                    send(id, frame);
                    success = false;
                }
            }
            if (success)
                break;
            std::this_thread::sleep_for(std::chrono::microseconds(response_wait_microsecond));
        }
        fmt::print(success ? fmt_tyle_ok() : fmt_style_error(), "[{0:%F %T}][{1}:{2}][{3}]: motors enable {4} after {5:d} try\033[0m\n",
                   get_time_now(), __FILE__, __LINE__, __func__, success ? "success" : "failed", try_count);
        if (!success)
            for (int id = 0; id < num_motors; id++)
                if (motor_enable_mask(id))
                    fmt::print(target_status == mode_status(id) ? fmt_tyle_ok() : fmt_style_error(), "  motor#{:02d}={}\n", id, mode_status(id));
    }

    // Enable all motors (convenience wrapper)
    void enable() { enable(Eigen::Matrix<bool, Eigen::Dynamic, 1>::Constant(num_motors, true)); }

    // disable all motors
    void disable(bool clear_fault = false)
    {
        std::string status_str = "disable";
        MotorModeStatus expected_value = MotorModeStatus::Reset;
        // set to unknown so that the frist loop will be executed 
        mode_status.setConstant(MotorModeStatus::Unknown);
        bool success;
        int try_count=-1;
        while (++try_count <= max_try_send)
        {   
            success = true;
            for (int id = 0; id < num_motors; id++)
            {
                if (mode_status(id) != expected_value)
                {
                    CanMessage frame;
                    frame.msgDisableMotor(master_can_id, can_ids(id),clear_fault);
                    send(id, frame);
                    success = false;
                }
            }
            if (success)
                break;
            std::this_thread::sleep_for(std::chrono::microseconds(response_wait_microsecond));
        }
        fmt::print(success ? fmt_tyle_ok() : fmt_style_error(), "[{0:%F %T}][{1}:{2}][{3}]: motors {4} {5} after {6:d} try\033[0m\n",
                   get_time_now(), __FILE__, __LINE__, __func__, status_str, success ? "success" : "failed", try_count);
        if (!success)
            for (int id = 0; id < num_motors; id++)
                fmt::print(expected_value == mode_status(id) ? fmt_tyle_ok() : fmt_style_error(), "  motor#{:02d}={}\n", id, mode_status(id));
    }

    // set all motors current position to zero
    void set_pos_zero(bool save_to_eeprom = false)
    {
        std::string status_str = "set_pos_zero";
        mech_pos.setConstant(0.02f);
        float tolerance =0.002;
        float expected_value = 0;

        bool success;
        int try_count=-1;
        while (++try_count <= max_try_send)
        {   
            success = true;
            for (int id = 0; id < num_motors; id++)
            {

                if (std::abs(mech_pos(id) - expected_value)>tolerance)
                {
                    CanMessage frame;
                    frame.msgSetPosZero(master_can_id, can_ids(id));
                    send(id, frame);
                    success = false;
                }else if (save_to_eeprom){
                    CanMessage frame;
                    frame.msgSaveData(master_can_id, can_ids(id));
                    send(id, frame);
                }
            }
            if (success && try_count > 0)
                break;
            std::this_thread::sleep_for(std::chrono::microseconds(response_wait_microsecond));
        }
        fmt::print(success ? fmt_tyle_ok() : fmt_style_error(), "[{0:%F %T}][{1}:{2}][{3}]: motors {4} {5} after {6:d} try\033[0m\n",
                   get_time_now(), __FILE__, __LINE__, __func__, status_str, success ? "success" : "failed", try_count);
        if (!success)
            for (int id = 0; id < num_motors; id++)
                fmt::print(std::abs(mech_pos(id) - expected_value)<=tolerance ? fmt_tyle_ok() : fmt_style_error(), "  motor#{:02d}={:.3f}\n", id, mech_pos(id));

    }

    // save motor parameters to flash memory
    void save_data()
    {
        for (int id : round_robin_motor_order)
        {
            CanMessage frame;
            frame.msgSaveData(master_can_id, can_ids(id));
            send(id, frame);
            std::this_thread::sleep_for(std::chrono::microseconds(response_wait_microsecond));
        }
        fmt::print(fmt_tyle_ok(), "[{0:%F %T}][{1}:{2}][{3}]: save_data command sent to all motors\033[0m\n",
                   get_time_now(), __FILE__, __LINE__, __func__);
    }

private:

    // Per-bus send queue: decouples the control thread from CAN socket writes,
    // allowing sends on different buses to happen in parallel.
    struct BusSendContext {
        std::queue<CanMessage> queue;
        std::mutex mutex;
        std::condition_variable cv;
        double last_send_time = 0.0;
    };
    std::vector<std::unique_ptr<BusSendContext>> bus_send_contexts;

    // Staging buffers for motion_control_once(): one per bus.
    // Pre-allocated members (not locals) so capacity is reused each cycle.
    std::vector<std::vector<CanMessage>> staged_frames;

    std::vector<int> round_robin_motor_order;

    void build_round_robin_order() {
        std::map<uint8_t, std::vector<int>> iface_to_ids;
        for (int i = 0; i < num_motors; ++i)
            iface_to_ids[can_interface_ids(i)].push_back(i);

        size_t max_len = 0;
        for (const auto& [_, group] : iface_to_ids)
            max_len = std::max(max_len, group.size());

        for (size_t k = 0; k < max_len; ++k) {
            for (size_t iface = 0; iface < can_interfaces.size(); ++iface) {
                auto it = iface_to_ids.find(iface);
                if (it != iface_to_ids.end() && k < it->second.size())
                    round_robin_motor_order.push_back(it->second[k]);
            }
        }
    }

    // Enqueue frame to the per-bus sender thread (returns immediately, non-blocking)
    inline void send(int i, CanMessage &frame)
    {
        int bus = can_interface_ids(i);
        {
            std::lock_guard<std::mutex> lock(bus_send_contexts[bus]->mutex);
            bus_send_contexts[bus]->queue.push(frame);
        }
        bus_send_contexts[bus]->cv.notify_one();
    }

    void receive_frame_thread(int i)
    {
        // Set RT priority for receiver thread (requires root/CAP_SYS_NICE)
        struct sched_param param;
        param.sched_priority = 90;
        if (sched_setscheduler(0, SCHED_FIFO, &param) != 0) {
            fmt::print(fmt_tyle_warn(), "[{0:%F %T}][{1}:{2}][{3}]: Failed to set RT priority for receiver #{4:d}: {5}\033[0m\n",
                       get_time_now(), __FILE__, __LINE__, __func__, i, strerror(errno));
            fmt::print(fmt_tyle_warn(), "  → Running without RT scheduling (may have delayed CAN feedback)\n");
            fmt::print(fmt_tyle_warn(), "  → To enable: run with sudo OR setcap cap_sys_nice=+ep <program>\033[0m\n");
        }

        std::string can_interface = can_interfaces[i];
        int socket_fd = socket_fds(i);

        struct can_frame frame;
        struct pollfd fds;
        fds.fd = socket_fd;
        fds.events = POLLIN;

        const int max_try_recv = 5;
        // timeout_recv_ms is read from the atomic member each iteration so Python can tighten
        // it after start_motion_control_continuously() without restarting the thread.
        int try_count = 0;
        while (receive_thread_should_run)
        {
            int ret = poll(&fds, 1, recv_timeout_ms); // re-read each iteration so Python can update it
            if (ret > 0)
            {
                if (fds.revents & POLLIN)
                {
                    int nbytes = read(socket_fd, &frame, sizeof(frame));
                    if (nbytes > 0)
                    {
                        try_count = 0;
                        {
                            std::lock_guard<std::mutex> lock(mutex_recv);
                            queue_recv.push(frame);
                        }
                        parseFrame(frame); // HACK to parse the frame immediately
                    }
                }
            }
            else if (receive_thread_should_run == false)
            {
                break;
            }
            else{
                try_count++;
                if (try_count >= max_try_recv){
                    if  (ret == 0){
                        fmt::print(fmt_tyle_warn(), "[{0:%F %T}][{1}:{2}][{3}]: Timeout waiting for CAN frame on {4}.\033[0m\n", get_time_now(), __FILE__, __LINE__, __func__, can_interface.c_str());
                    }else{
                        fmt::print(fmt_tyle_warn(), "[{0:%F %T}][{1}:{2}][{3}]: poll failed on {4}: {5}\033[0m\n", get_time_now(), __FILE__, __LINE__, __func__, can_interface.c_str(), strerror(errno));
                    }
                    // receive_thread_should_run = false;
                    break;
                }


            }
        }
        receive_thread_should_run = false;
        fmt::print(fmt_tyle_ok(), "[{0:%F %T}][{1}:{2}][{3}]: receiving thread on {4} exit success.\033[0m\n", get_time_now(), __FILE__, __LINE__, __func__, can_interface.c_str());
    }

    void send_frame_thread(int bus_id)
    {
        // Set RT priority (same as receiver threads)
        struct sched_param param;
        param.sched_priority = 90;
        if (sched_setscheduler(0, SCHED_FIFO, &param) != 0) {
            fmt::print(fmt_tyle_warn(), "[{0:%F %T}][{1}:{2}][{3}]: Failed to set RT priority for sender #{4}: {5}\033[0m\n",
                       get_time_now(), __FILE__, __LINE__, __func__, bus_id, strerror(errno));
        }

        auto& ctx = *bus_send_contexts[bus_id];

        while (true) {
            CanMessage msg;
            {
                std::unique_lock<std::mutex> lock(ctx.mutex);
                // Wait until there's a frame to send or shutdown is requested
                ctx.cv.wait(lock, [&]{ return !ctx.queue.empty() || !receive_thread_should_run; });
                if (ctx.queue.empty()) break;  // shutdown with empty queue — done
                msg = std::move(ctx.queue.front());
                ctx.queue.pop();
            }

            // Enforce minimum inter-frame gap on this bus (busy wait for sub-ms precision)
            {
                auto now = std::chrono::steady_clock::now();
                double now_sec = std::chrono::duration<double>(now.time_since_epoch()).count();
                double elapsed_us = (now_sec - ctx.last_send_time) * 1e6;
                if (elapsed_us < min_same_bus_delay_us) {
                    auto wait_until = now + std::chrono::microseconds(
                        static_cast<int>(min_same_bus_delay_us - elapsed_us));
                    while (std::chrono::steady_clock::now() < wait_until);  // busy wait
                }
            }

            // Send with retries on EAGAIN (kernel TX buffer full)
            constexpr int max_retries = 10;
            constexpr int retry_delay_us = 50;
            for (int retry = 0; retry < max_retries && receive_thread_should_run; retry++) {
                if (msg.send(socket_fds(bus_id), should_print_send)) {
                    ctx.last_send_time = std::chrono::duration<double>(
                        std::chrono::steady_clock::now().time_since_epoch()).count();
                    break;
                }
                std::this_thread::sleep_for(std::chrono::microseconds(retry_delay_us));
            }
        }
        fmt::print(fmt_tyle_ok(), "[{0:%F %T}][{1}:{2}][{3}]: sender thread #{4} exit success.\033[0m\n",
                   get_time_now(), __FILE__, __LINE__, __func__, bus_id);
    }
};

void ParseMotorData(can_frame &frame, const std::map<uint8_t, MotorSpec> &motor_spec_map);

#endif
