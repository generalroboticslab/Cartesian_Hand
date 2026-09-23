#include "gl_motor.hpp"

#include <cstring>
#include <cerrno>
#include <unistd.h>
#include <net/if.h>
#include <sys/socket.h>

// GL40 II default scaling. Update alongside the GLMotorType enum.
std::map<uint8_t, MotorSpec> gl_spec_map = {
    {GL40, gl_spec_gl40},
};

// ################### GLCanMessage ###########################################
GLCanMessage::GLCanMessage() : can_frame{.len = 8} {}

void GLCanMessage::msgMotionControl(const uint8_t &node_id, const uint16_t &pos, const uint16_t &vel,
                                    const uint16_t &kp, const uint16_t &kd, const uint16_t &torque)
{
    // MIT mode (0): CAN ID is the node id. Standard frame — no EFF flag.
    this->can_id = node_id;
    this->len = 8;
    // Bit packing per spec §4.1: pos 16-bit, vel/kp/kd/torque 12-bit.
    data[0] = (pos >> 8) & 0xFF;
    data[1] = pos & 0xFF;
    data[2] = (vel >> 4) & 0xFF;
    data[3] = (((vel & 0x0F) << 4) | ((kp >> 8) & 0x0F)) & 0xFF;
    data[4] = kp & 0xFF;
    data[5] = (kd >> 4) & 0xFF;
    data[6] = (((kd & 0x0F) << 4) | ((torque >> 8) & 0x0F)) & 0xFF;
    data[7] = torque & 0xFF;
}

void GLCanMessage::msgMotionPosVel(const uint8_t &node_id, const float &pos, const float &vel)
{
    // Position-Velocity mode (mode 1): CAN ID = 0x100 | node_id. Standard frame.
    // Payload = two little-endian IEEE-754 float32 (pos [rad], vel limit [rad/s]).
    this->can_id = ((uint32_t)GL_MODE_POS_VEL << 8) | node_id;
    this->len = 8;
    memcpy(&data[0], &pos, 4); // x86 is little-endian; floats copy directly
    memcpy(&data[4], &vel, 4);
}

void GLCanMessage::msgMotionVel(const uint8_t &node_id, const float &vel)
{
    // Velocity mode (mode 2): CAN ID = 0x200 | node_id. Payload = one LE float32 [rad/s].
    this->can_id = ((uint32_t)GL_MODE_VEL << 8) | node_id;
    this->len = 4;
    memcpy(&data[0], &vel, 4);
}

void GLCanMessage::msgCommand(const uint8_t &node_id, const uint8_t &mode, const uint8_t &suffix)
{
    // Config command: CAN ID = (mode << 8) | node_id, payload FF*7 + suffix.
    this->can_id = ((uint32_t)mode << 8) | node_id;
    this->len = 8;
    memset(data, 0xFF, 7);
    data[7] = suffix;
}

bool GLCanMessage::send(const int &socket_fd, bool should_print)
{
    // GL II is standard-frame only: do NOT set CAN_EFF_FLAG.
    if (write(socket_fd, static_cast<can_frame *>(this), sizeof(can_frame)) != (ssize_t)sizeof(can_frame))
    {
        // Don't print for EINTR (signal interrupt - likely Ctrl+C) or EAGAIN
        if (errno != EAGAIN && errno != EWOULDBLOCK && errno != EINTR)
        {
            struct sockaddr_can addr;
            socklen_t len = sizeof(addr);
            char ifname[IFNAMSIZ] = "unknown";
            if (getsockname(socket_fd, (struct sockaddr *)&addr, &len) == 0)
                if_indextoname(addr.can_ifindex, ifname);
            fmt::print(stderr, fmt_style_error(), "[{0:%F %T}][{1}:{2}][{3}]: Error sending CAN frame on {4} (fd={5}): {6}\033[0m\n",
                       get_time_now(), __FILE__, __LINE__, __func__, ifname, socket_fd, strerror(errno));
        }
        return false;
    }
    if (should_print)
    {
        fmt::print("[SENT][ID]: 0x{:x} Data:", this->can_id);
        for (int i = 0; i < this->len; i++)
            fmt::print(" {0:02x}", this->data[i]);
        fmt::print("\n");
    }
    return true;
}
