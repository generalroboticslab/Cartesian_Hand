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

#include <thread>
#include <queue>
#include <mutex>
#include <condition_variable>

#include "motor.hpp"
#include <cassert>
#include <iomanip>


std::map<uint8_t, MotorSpec> motor_spec_map = {
    {R00, motor_spec_r00},
    {R01, motor_spec_r01}, 
    {R02, motor_spec_r02}, 
    {R03, motor_spec_r03}, 
    {R04, motor_spec_r04},
    {R05, motor_spec_r05},
    {R06, motor_spec_r06},
    };

    
// std::map<uint8_t, std::string> MotorCommunicationTypeToStringMap ={
//     {GetCanID, "GetCanID"},
//     {MotionControl, "MotionControl"},
//     {MotorStatus, "MotorStatus"},
//     {Enable, "Enable"},
//     {Disable, "Disable"},
//     {SetPosZero, "SetPosZero"},
//     {ChangeCanID, "ChangeCanID"},
//     {ControlMode, "ControlMode"},
//     {GetSingleParameter, "GetSingleParameter"},
//     {SetSingleParameter, "SetSingleParameter"},
//     {Error, "Error"},
// };

// std::map<uint16_t, std::string> MotorParamsToStringMap = {
//     {run_mode, "run_mode"},
//     {iq_ref, "iq_ref"},
//     {mech_vel_ref, "mech_vel_ref"},
//     {torque_limit, "torque_limit"},
//     {cur_kp, "cur_kp"},
//     {cur_ki, "cur_ki"},
//     {cur_filt_gain, "cur_filt_gain"},
//     {mech_pos_ref, "mech_pos_ref"},
//     {spd_limit, "spd_limit"},
//     {cur_limit, "cur_limit"},
//     {mech_pos, "mech_pos"},
//     {iqf, "iqf"},
//     {mech_vel, "mech_vel"},
//     {v_bus, "v_bus"},
//     {loc_kp, "loc_kp"},
//     {spd_kp, "spd_kp"},
//     {spd_ki, "spd_ki"},
//     {spd_filt_gain, "spd_filt_gain"},
// };


bool setup_socket(const std::string &interface, int &socket_fd)
{
    struct sockaddr_can addr;
    struct ifreq ifr;

    // Create socket
    socket_fd = socket(PF_CAN, SOCK_RAW, CAN_RAW);
    // Increase socket send buffer size to 1MB
    int buf_size = 1024 * 1024;
    if (setsockopt(socket_fd, SOL_SOCKET, SO_SNDBUF, &buf_size, sizeof(buf_size)) < 0) {
        fmt::print(stderr, fmt_style_error(), "[{0:%F %T}][{1}:{2}][{3}]: Failed to set SO_SNDBUF: {4}\033[0m\n",
                get_time_now(), __FILE__, __LINE__, __func__, strerror(errno));
    }

    if (socket_fd < 0)
    {
        fmt::print(stderr, fmt_style_error(), "[{0:%F %T}][{1}:{2}][{3}]: Error creating CAN socket: {4}\033[0m\n", get_time_now(), __FILE__, __LINE__, __func__, strerror(errno));
        return false;
    }

    // Set CAN interface
    std::strncpy(ifr.ifr_name, interface.c_str(), IFNAMSIZ - 1);
    if (ioctl(socket_fd, SIOCGIFINDEX, &ifr) < 0)
    {
        fmt::print(stderr, fmt_style_error(), "[{0:%F %T}][{1}:{2}][{3}]: Error getting CAN interface index: {4}\033[0m\n", get_time_now(), __FILE__, __LINE__, __func__, strerror(errno));
        return false;
    }

    addr.can_family = AF_CAN;
    addr.can_ifindex = ifr.ifr_ifindex;

    // Check if network interface is up
    if (ioctl(socket_fd, SIOCGIFFLAGS, &ifr) < 0)
    {
        fmt::print(stderr, fmt_style_error(), "[{0:%F %T}][{1}:{2}][{3}]: Error getting interface flags: {4}\033[0m\n", get_time_now(), __FILE__, __LINE__, __func__, strerror(errno));
        close(socket_fd);
        return false;
    }
    if (!(ifr.ifr_flags & IFF_UP))
    {
        fmt::print(stderr, fmt_style_error(), "[{0:%F %T}][{1}:{2}][{3}]: Network interface {4} is down (make sure to bring it up with 'sudo ip link set {4} up' before running this program)\033[0m\n", get_time_now(), __FILE__, __LINE__, __func__, interface.c_str());
        close(socket_fd);
        return false;
    }

    // Bind socket to CAN interface
    if (bind(socket_fd, (struct sockaddr *)&addr, sizeof(addr)) < 0)
    {
        fmt::print(stderr, fmt_style_error(), "[{0:%F %T}][{1}:{2}][{3}]: Error binding CAN socket: {4}\033[0m\n", get_time_now(), __FILE__, __LINE__, __func__, strerror(errno));
        return false;
    }

    // Set non-blocking mode for clean shutdown on Ctrl+C
    int flags = fcntl(socket_fd, F_GETFL, 0);
    if (flags == -1) {
        fmt::print(stderr, fmt_style_error(), "[{0:%F %T}][{1}:{2}][{3}]: Error getting socket flags: {4}\033[0m\n",
                get_time_now(), __FILE__, __LINE__, __func__, strerror(errno));
        return false;
    }
    if (fcntl(socket_fd, F_SETFL, flags | O_NONBLOCK) == -1) {
        fmt::print(stderr, fmt_style_error(), "[{0:%F %T}][{1}:{2}][{3}]: Error setting non-blocking mode: {4}\033[0m\n",
                get_time_now(), __FILE__, __LINE__, __func__, strerror(errno));
        return false;
    }

    // Success
    fmt::print(fmt_tyle_ok(), "[{0:%F %T}][{1}:{2}][{3}]: {4} Setup success.\033[0m\n", get_time_now(), __FILE__, __LINE__, __func__,interface.c_str());
    return true;
}

void send_extended_frame(int socket_fd, uint32_t can_id, const uint8_t *data, uint8_t len)
{
    struct can_frame frame;
    frame.can_id = can_id | CAN_EFF_FLAG; // Set extended frame format flag
    frame.can_dlc = len;
    std::memcpy(frame.data, data, len);

    if (write(socket_fd, &frame, sizeof(frame)) != sizeof(frame))
    {
        fmt::print(stderr, fmt_style_error(), "[{0:%F %T}][{1}:{2}][{3}]: Error sending CAN frame to {4}: {5}\033[0m\n", get_time_now(), __FILE__, __LINE__, __func__, socket_fd, strerror(errno));
    }
    else
    {
        fmt::print("[SENT] CAN [ID]: 0x{:x} Data:", frame.can_id);
        for (int i = 0; i < frame.can_dlc; i++)
        {
            fmt::print(" 0x{:x}", frame.data[i]);
        }
        fmt::print("\n");
    }
}

void print_frame(can_frame &frame, std::string header) {
    fmt::print("{:s}[ID]: 0x{:x} Data:",header, frame.can_id);
    for (int i = 0; i < frame.len; i++)
    {
        fmt::print(" {:02x}", frame.data[i]);
    }
    fmt::print("\n");
}


// void receive_frame_thread(int socket_fd) {
//     struct can_frame frame;
//     struct pollfd fds;
//     fds.fd = socket_fd;
//     fds.events = POLLIN;

//     while (receive_thread_should_run) {
//         int ret = poll(&fds, 1, 200); // Timeout after 200 ms
//         if (ret > 0) {
//             if (fds.revents & POLLIN) {
//                 int nbytes = read(socket_fd, &frame, sizeof(frame));
//                 if (nbytes > 0) {
//                     {
//                         std::lock_guard<std::mutex> lock(mutex_recv);
//                         queue_recv.push(frame);
//                     }
//                     cv_recv.notify_one(); // Notify the main thread
//                 }
//             }
//         } else if (ret == 0) {
//             // Optional: You might not need this in a separate thread
//             std::cout << "Timeout waiting for CAN frame" << std::endl;
//         } else {
//             perror("poll failed");
//             break; // Exit the loop on error
//         }
//     }
// }

// ################### CANMessage ############################################
CanMessage::CanMessage() : can_frame{.len = 8} {}

void CanMessage::msgChangeCanId(const uint8_t &master_can_id, const uint8_t &target_can_id, const uint8_t &new_target_can_id){
    this->can_id = MotorCommunicationType::ChangeCanID << 24 | new_target_can_id<<16 | master_can_id << 8 | target_can_id;
    // memset(data, 0, sizeof(data));
}


void CanMessage::msgGetDeviceMcuId(const uint8_t &master_can_id, const uint8_t &target_can_id)
{
    this->can_id = MotorCommunicationType::GetCanID << 24 | master_can_id << 8 | target_can_id;
    memset(data, 0, sizeof(data));
}

void CanMessage::msgEnableMotor(const uint8_t &master_can_id, const uint8_t &target_can_id)
{
    this->can_id = MotorCommunicationType::Enable << 24 | master_can_id << 8 | target_can_id;
    memset(data, 0, sizeof(data));
}

void CanMessage::msgDisableMotor(const uint8_t &master_can_id, const uint8_t &target_can_id,bool clear_fault)
{
    this->can_id = MotorCommunicationType::Disable << 24 | master_can_id << 8 | target_can_id;
    memset(data, 0, sizeof(data));
    data[0] = (uint8_t)clear_fault; // clear fault
}

void CanMessage::msgMotionControl(const uint8_t &target_can_id, const uint16_t &torque, const uint16_t &pos, const uint16_t &vel, const uint16_t &kp, const uint16_t &kd)
{
    this->can_id = MotorCommunicationType::MotionControl << 24 | torque << 8 | target_can_id;
    // memset(data, 0, sizeof(data));
    data[0] = pos >> 8;
    data[1] = pos;
    data[2] = vel >> 8;
    data[3] = vel;
    data[4] = kp >> 8;
    data[5] = kp;
    data[6] = kd >> 8;
    data[7] = kd;
}

void CanMessage::msgSetPosZero(const uint8_t &master_can_id, const uint8_t &target_can_id)
{
    this->can_id = MotorCommunicationType::SetPosZero << 24 | master_can_id << 8 | target_can_id;
    memset(data, 0, sizeof(data));
    data[0] = 1;
}

void CanMessage::msgReadParam(const uint8_t &master_can_id, const uint8_t &target_can_id, const uint16_t &motor_param)
{
    this->can_id = MotorCommunicationType::GetSingleParameter << 24 | master_can_id << 8 | target_can_id;
    memset(data, 0, sizeof(data));
    data[0] = motor_param;
    data[1] = motor_param >> 8;
    // std::cout << "Data: " << std::hex << (int)data[0] << " " << (int)data[1] << std::dec << std::endl;
}

void CanMessage::msgSaveData(const uint8_t &master_can_id, const uint8_t &target_can_id)
{
    this->can_id = MotorCommunicationType::SaveData << 24 | master_can_id << 8 | target_can_id;
    data[0] = 0x01; data[1] = 0x02; data[2] = 0x03; data[3] = 0x04;
    data[4] = 0x05; data[5] = 0x06; data[6] = 0x07; data[7] = 0x08;
}

bool CanMessage::send(const int &socket_fd, bool should_print)
{
    this->can_id = this->can_id | CAN_EFF_FLAG;
    // this->len = 8; // Set payload length, always 8

    if (write(socket_fd, this, sizeof(*this)) != sizeof(*this))
    {
        // Don't print for EINTR (signal interrupt - likely Ctrl+C) or EAGAIN
        if (errno != EAGAIN && errno != EWOULDBLOCK && errno != EINTR) {
            // Get interface name from socket fd
            struct sockaddr_can addr;
            socklen_t len = sizeof(addr);
            char ifname[IFNAMSIZ] = "unknown";
            if (getsockname(socket_fd, (struct sockaddr *)&addr, &len) == 0) {
                if_indextoname(addr.can_ifindex, ifname);
            }
            fmt::print(stderr, fmt_style_error(), "[{0:%F %T}][{1}:{2}][{3}]: Error sending CAN frame on {4} (fd={5}): {6}\033[0m\n", get_time_now(), __FILE__, __LINE__, __func__, ifname, socket_fd, strerror(errno));
        }
        return false;
    }
    if (should_print)
    {
        // fmt::print("[{0:%F %T}][{1}:{2}]:", get_time_now(), __FILE__, __LINE__, __func__ );
        fmt::print("[SENT][ID]: 0x{:x} Data:", this->can_id);
        for (int i = 0; i < this->len; i++)
        {
            fmt::print(" {0:02x}", this->data[i]);
        }
        fmt::print("\n");
    }
    return true;
}

// ################### CanController ############################################

// CanController::CanController(std::string can_interface)
// {
//     this->can_interface = can_interface;
// }

CanController::~CanController()
{
    receive_thread_should_run = false;
    if (started)
    {
        receiver_thread.join();
        close(socket_fd);
        fmt::print(fmt_tyle_ok(), "[{0:%F %T}][{1}:{2}][{3}]: joining thread success.\n", get_time_now(), __FILE__, __LINE__, can_interface.c_str());
    }
    fmt::print(fmt_tyle_ok(), "[{0:%F %T}][{1}:{2}][{3}]: shutdown success.\033[0m\n", get_time_now(), __FILE__, __LINE__, can_interface.c_str());
}

void CanController::CanController::start()
{
    setup_socket(can_interface, socket_fd); // setup the socket

    // Start the receiver thread
    receiver_thread = std::thread(&CanController::receive_frame_thread, this);
    started = true;
}

void CanController::send(can_frame &frame)
{
    frame.can_id = frame.can_id | CAN_EFF_FLAG;
    frame.len = 8; // Set payload length, always 8

    if (write(socket_fd, &frame, sizeof(frame)) != sizeof(frame))
    {
        fmt::print(stderr, "CAN frame send failed: {}\n", strerror(errno));
    }
    else
    {
        counter_sent += 1; // Use the global counter_sent

        fmt::print("[SENT][ID]: 0x{:x} Data: ", frame.can_id);
        for (int i = 0; i < frame.len; i++)
        {
            fmt::print(" 0x{:x}", frame.data[i]);
        }
        fmt::print("\n");
    }
}
/*receive 1 frame*/
can_frame CanController::receive(int timeout_millisecond,int wait_microsecond){
    
    can_frame frame{};
    auto start_time = std::chrono::high_resolution_clock::now();
    do{
        // Check if there is data available without waiting
        if (!queue_recv.empty())
        {
            std::unique_lock<std::mutex> lock(mutex_recv);
            frame = std::move(queue_recv.front());
            queue_recv.pop();
            counter_received += 1;
            return frame;
        }
        else
        {
            std::this_thread::sleep_for(std::chrono::microseconds(wait_microsecond));
        }
    }
    while ((std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::high_resolution_clock::now() - start_time)).count() < timeout_millisecond);
    fmt::print(fmt_tyle_warn(), "[{0:%F %T}][{1}:{2}][{3}]: Timeout waiting for CAN frame, exiting.\033[0m\n", get_time_now(), __FILE__, __LINE__, can_interface.c_str());
    return frame;
}


void CanController::receive_frame_thread()
{
    struct can_frame frame;
    struct pollfd fds;
    fds.fd = this->socket_fd;
    fds.events = POLLIN;

    while (receive_thread_should_run)
    {
        int ret = poll(&fds, 1, 200); // Timeout after 200 ms
        if (ret > 0)
        {
            if (fds.revents & POLLIN)
            {
                int nbytes = read(socket_fd, &frame, sizeof(frame));
                if (nbytes > 0)
                {
                    {
                        std::lock_guard<std::mutex> lock(mutex_recv);
                        queue_recv.push(frame);
                    }
                }
            }
        }
        else if (ret == 0)
        {
            fmt::print(fmt_tyle_warn(), "[{0:%F %T}][{1}:{2}][{3}]: Timeout waiting for CAN frame, exiting.\033[0m\n", get_time_now(), __FILE__, __LINE__, can_interface.c_str());
            break;
        }
        else
        {
            fmt::print(stderr, fmt_style_error(), "[{0:%F %T}][{1}:{2}][{3}]: Error receiving CAN frame: {4}\033[0m\n", get_time_now(), __FILE__, __LINE__, __func__, strerror(errno));
            break;
        }
    }
    receive_thread_should_run = false;
    fmt::print(fmt_tyle_ok(), "[{0:%F %T}][{1}:{2}][{3}]: receiving thread on {4} exited.\033[0m\n", get_time_now(), __FILE__, __LINE__, __func__, can_interface.c_str());
}

// ############################################################################

/*******************************************************************************
 * @Function        : Convert a uint16_t integer to a float
 * @Parameter 1     : The uint16_t value to convert
 * @Parameter 2     : Minimum value of the float range
 * @Parameter 3     : Maximum value of the float range
 * @Parameter 4     : Number of bits used in the integer
 * @ReturnValue     : The converted float value
 *******************************************************************************/
float uint16_to_float(uint16_t x, float x_min, float x_max, int bits)
{
    uint32_t span = (1 << bits) - 1;  // Calculate the total possible values in the integer range
    float offset = x_max - x_min;     // Calculate the span of the float range
    return offset * x / span + x_min; // Map the integer to the corresponding float value
}

/*******************************************************************************
 * @Function        : Convert a float to an integer
 * @Parameter 1     : The float value to convert
 * @Parameter 2     : Minimum value of the float range
 * @Parameter 3     : Maximum value of the float range
 * @Parameter 4     : Number of bits used in the output integer
 * @ReturnValue     : The converted integer value
 *******************************************************************************/
int float_to_uint(float x, float x_min, float x_max, int bits)
{
    float span = x_max - x_min; // Calculate the span of the float range
    float offset = x_min;       // Get the minimum value of the float range
    if (x > x_max)
        x = x_max; // Clamp the float value to the maximum if it exceeds it
    else if (x < x_min)
        x = x_min;                                                  // Clamp the float value to the minimum if it's below it
    return (int)((x - offset) * ((float)((1 << bits) - 1)) / span); // Map the float to the corresponding integer value
}

uint16_t float_to_uint16(float x, const float &x_min, const float &x_max)
{
    // Clamp to range
    if (x > x_max) x = x_max;
    else if (x < x_min) x = x_min;

    // Map [x_min, x_max] → [0, 65535]
    constexpr float max_uint16 = 65535.0f;
    float normalized = (x - x_min) / (x_max - x_min);
    return static_cast<uint16_t>(normalized * max_uint16);
}

// TODO: THIS IS NOT WORKING YET FOR POS CONTROL. FIXME
uint16_t float_to_uint16_loop(float x, const float &x_min, const float &x_max)
{
    float span = x_max - x_min; // Calculate the span of the float range
    float offset = x_min;       // Get the minimum value of the float range
    if (x > x_max)
        x = fmod(x,span);
        // x = x_max; // Clamp the float value to the maximum if it exceeds it
    else if (x < x_min)
        x = fmod(x,span);
        // x = x_min;                                                // Clamp the float value to the minimum if it's below it
    return (int)((x - offset) * ((float)((1 << 16) - 1)) / span); // Map the float to the corresponding integer value
}

/*******************************************************************************
 * @Function         : Convert a byte array (uint8_t*) to a float
 * @Parameter        : The byte array containing the float data
 * @ReturnValue      : The converted float value
 *******************************************************************************/
float byte_to_float(uint8_t *bytedata)
{
    union {
        uint8_t bytes[4];
        float f;
    } data;
    // data.bytes[0] = bytedata[4];
    // data.bytes[1] = bytedata[5];
    // data.bytes[2] = bytedata[6];
    // data.bytes[3] = bytedata[7];
    memcpy(data.bytes, bytedata + 4, sizeof(data.bytes));
    return data.f;

    // uint32_t data = bytedata[7] << 24 | bytedata[6] << 16 | bytedata[5] << 8 | bytedata[4]; // Combine the bytes into a 32-bit integer
    // float data_float;
    // memcpy(&data_float, &data, sizeof(data_float)); // Reinterpret the integer as a float
    // return data_float;                              // Return the float value
}

/*******************************************************************************
 * @Function         : Convert a byte array (uint8_t*) to a uint16_t
 * @Parameter        : The byte array containing the uint16_t data
 * @ReturnValue      : The converted uint16_t value
 *******************************************************************************/
uint16_t byte_to_uint16(uint8_t *bytedata)
{
    uint16_t data = bytedata[5] << 8 | bytedata[4]; // Combine the bytes into a 16-bit integer
    return data;                                    // Return the 16-bit integer
}

uint32_t byte_to_uint32(uint8_t *bytedata)
{
    union {
        uint8_t bytes[4];
        uint32_t f;
    } data;
    // data.bytes[0] = bytedata[4];
    // data.bytes[1] = bytedata[5];
    // data.bytes[2] = bytedata[6];
    // data.bytes[3] = bytedata[7];
    memcpy(data.bytes, bytedata + 4, sizeof(data.bytes));
    return data.f;
    // uint32_t data = bytedata[7] << 24 | bytedata[6] << 16 | bytedata[5] << 8 | bytedata[4]; // Combine the bytes into a 32-bit integer
    // return data;
}



// ############################################################################
/******************* CanMotorController **************************************/

// CanMotorController::CanMotorController(
//     std::vector<std::string> can_interface,
//     Eigen::Matrix<uint8_t, Eigen::Dynamic, 1> can_interface_ids,
//     Eigen::Matrix<uint8_t, Eigen::Dynamic, 1> can_ids,
//     Eigen::Matrix<uint8_t, Eigen::Dynamic, 1> motor_types)
// {
//     this->num_motors = can_ids.size();
//     this->can_ids = can_ids;
//     this->motor_types = motor_types;
// }

// ############################################################################

void ParseMotorData(can_frame &frame, const std::map<uint8_t, MotorSpec> &motor_spec_map)
{

    uint8_t communication_type = (frame.can_id & 0x1F000000) >> 24;
    switch (communication_type)
    {
    case MotorCommunicationType::GetCanID:
    {
        // mcu ids
        uint8_t target_can_id = (frame.can_id & 0xFFFF00) >> 8;
        uint8_t least_id_byte = (frame.can_id & 0xFF);
        assert(least_id_byte == 0xFE);
        uint64_t mcu_id;
        std::memcpy(&mcu_id, frame.data, sizeof(mcu_id));
        fmt::print("[RECV] mcu id: 0x{0:x}\n", mcu_id);
        break;
    }

    case MotorCommunicationType::MotorStatus:
    {
        // uint8_t master_can_id = (frame.can_id & 0xFF);
        uint8_t target_can_id = (frame.can_id & 0xFF00) >> 8;

        auto &spec = motor_spec_map.at(target_can_id);

        float pos = uint16_to_float(frame.data[0] << 8 | frame.data[1], spec.pos_min, spec.pos_max, 16);
        float vel = uint16_to_float(frame.data[2] << 8 | frame.data[3], spec.vel_min, spec.vel_max, 16);
        float torque = uint16_to_float(frame.data[4] << 8 | frame.data[5], spec.torque_min, spec.torque_max, 16);
        float temp = (float)(frame.data[6] << 8 | frame.data[7]) * 0.1f;
        uint8_t error_code = uint8_t((frame.can_id & 0x3F0000) >> 16);
        uint8_t pattern = uint8_t((frame.can_id & 0xC00000) >> 22); // 0: reset, 1: calibration, 2: running

        fmt::print("[RECV] motor status: can_id: 0x{:x} pos: {:.3f} rad, vel: {:.2f} rad/s, torque: {:.1f} Nm, temp: {:.1f} °C, error_code: {}, pattern: {}\n",
                   target_can_id, pos, vel, torque, temp, error_code, pattern);
        break;
    }
    case MotorCommunicationType::GetSingleParameter:
    {
        uint16_t motor_param = frame.data[1] << 8 | frame.data[0];
        fmt::print("[RECV] motor param: 0x{:04x}:  ", motor_param);

        // assert (motor_param == 0x7005);
        switch (motor_param)
        {
        case MotorParams::run_mode:
        {
            uint8_t run_mode = frame.data[4];
            fmt::print("[RECV] motor run mode: 0x{:x}\n", run_mode);
            break;
        }
        case MotorParams::iq_ref:
        {
            float iq_ref = byte_to_float(frame.data);
            fmt::print("[RECV] iq ref: {:.2f}\n", iq_ref);
            break;
        }
        case MotorParams::mech_vel_ref:
        {
            float mech_vel_ref = byte_to_float(frame.data);
            fmt::print("[RECV] spd ref: {:.2f}\n", mech_vel_ref);
            break;
        }
        case MotorParams::torque_limit:
        {
            float torque_limit = byte_to_float(frame.data);
            fmt::print("[RECV] torque limit: {:.2f}\n", torque_limit);
            break;
        }
        case MotorParams::cur_kp:
        {
            float cur_kp = byte_to_float(frame.data);
            fmt::print("[RECV] cur kp: {:.2f}\n", cur_kp);
            break;
        }
        case MotorParams::cur_ki:
        {
            float cur_ki = byte_to_float(frame.data);
            fmt::print("[RECV] cur ki: {:.2f}\n", cur_ki);
            break;
        }
        case MotorParams::cur_filt_gain:
        {
            float cur_filt_gain = byte_to_float(frame.data);
            break;
        }
        case MotorParams::mech_pos_ref:
        {
            float mech_pos_ref = byte_to_float(frame.data);
            fmt::print("[RECV] loc ref: {:.2f}\n", mech_pos_ref);
            break;
        }
        case MotorParams::spd_limit:
        {
            float spd_limit = byte_to_float(frame.data);
            fmt::print("[RECV] spd limit: {:.2f}\n", spd_limit);
            break;
        }
        case MotorParams::cur_limit:
        {
            float cur_limit = byte_to_float(frame.data);
            fmt::print("[RECV] cur limit: {:.2f}\n", cur_limit);
            break;
        }
        case MotorParams::mech_pos:
        {
            float mech_pos = byte_to_float(frame.data);
            fmt::print("[RECV] mech pos: {:.2f}\n", mech_pos);
            break;
        }
        case MotorParams::iqf:
        {
            float iqf = byte_to_float(frame.data);
            fmt::print("[RECV] iqf: {:.2f}\n", iqf);
            break;
        }
        case MotorParams::mech_vel:
        {
            float mech_vel = byte_to_float(frame.data);
            fmt::print("[RECV] mech vel: {:.2f}\n", mech_vel);
            break;
        }
        case MotorParams::v_bus:
        {
            float v_bus = byte_to_float(frame.data);
            fmt::print("[RECV] v_bus: {:.2f}\n", v_bus);
            break;
        }
        // case MotorParams::rotation:
        // {
        //     int16_t rotation = frame.data[1] << 8 | frame.data[0];
        //     fmt::print("[RECV] rotation: %d\n", rotation);
        //     break;
        // }
        case MotorParams::loc_kp:
        {
            float loc_kp = byte_to_float(frame.data);
            fmt::print("[RECV] loc kp: {:.2f}\n", loc_kp);
            break;
        }
        case MotorParams::spd_kp:
        {
            float spd_kp = byte_to_float(frame.data);
            fmt::print("[RECV] spd kp: {:.2f}\n", spd_kp);
            break;
        }
        case MotorParams::spd_ki:
        {
            float spd_ki = byte_to_float(frame.data);
            fmt::print("[RECV] spd ki: {:.2f}\n", spd_ki);
            break;
        }
        case MotorParams::spd_filt_gain:
        {
            float spd_filt_gain = byte_to_float(frame.data);
            fmt::print("[RECV] spd filt gain: {:.2f}\n", spd_filt_gain);
            break;
        }
        default:
        {
            fmt::print("[RECV] {0:x}{1:x}{2:x}{3:x}{4:x}{5:x}{6:x}{7:x} unknown param\n",
                frame.data[0],frame.data[1],frame.data[2],frame.data[3],frame.data[4],frame.data[5],frame.data[6],frame.data[7]);
            break;
        }
            
        }
        break;
    }
    };
}
