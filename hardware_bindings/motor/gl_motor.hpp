#ifndef __GL_MOTOR_HPP__
#define __GL_MOTOR_HPP__

// CubeMars GL II gimbal motor (GL40 II, KV82.5) controller.
//
// This is a SEPARATE controller from CanMotorController (Robstride/CyberGear) on
// purpose: the GL II protocol shares nothing with Robstride at the wire level —
// standard 11-bit frames (not extended), a different MIT bit-packing, FF-prefixed
// config commands, feedback routed by node-id low-nibble, and no 0x70xx parameter
// model. GL40 also runs on its own dedicated CAN bus. The only things reused are the
// OS/threading helpers, which are free functions / standalone classes in motor.hpp.
//
// All protocol facts are from cubemars_reference/cubemars_gl_ii_can_spec.md
// (manual pp.41-49, hex-verified).

#include "motor.hpp" // reuse setup_socket, float_to_uint, uint16_to_float, LatencyLogger,
                      // FpsCounter, MotorSpec, MotorModeStatus, fmt_* helpers, RT/mlock pattern

// GL40 II default MIT scaling — must match the driver "Amplitude" configuration page.
// PMAX ±12.5, TMAX ±10, KP 0–500, KD 0–5 are hex-verified and unambiguous.
// ⚠ VMAX: manual self-contradicts (±200 protocol text vs ±250 speed hex example vs ±30
// PWM page / GLII.ino). ±200 is physically correct for KV82.5 @ 24V (~207 rad/s no-load).
// CONFIRM against the actual driver Amplitude setting on hardware before trusting vel.
constexpr MotorSpec gl_spec_gl40 = {-12.5f, 12.5f, -200.0f, 200.0f, 0.0f, 500.0f, 0.0f, 5.0f, -10.0f, 10.0f};

enum GLMotorType : uint8_t
{
    GL40 = 0,
}; // YOU MUST UPDATE gl_spec_map as well.

extern std::map<uint8_t, MotorSpec> gl_spec_map; // defined in gl_motor.cpp

// GL II control modes. The command CAN ID is (mode << 8) | node, and the motor only
// answers commands (incl. enter/exit/zero/clear) sent on its CONFIGURED mode's ID.
//   MIT (0):      pos/vel/kp/kd/tff impedance — requires MIT-enabled firmware.
//   POS_VEL (1):  cascaded position loop + velocity limit; two LE float32 (pos, vel_limit).
//   VEL (2):      pure velocity loop; one LE float32 (vel).
// Hardware note: the GL40 II tested (node 1) shipped in POS_VEL (1) and ignores MIT frames.
enum GLControlMode : uint8_t
{
    GL_MODE_MIT = 0,
    GL_MODE_POS_VEL = 1,
    GL_MODE_VEL = 2,
};

// GL II diagnostic/fault codes (upper nibble of feedback Data[0]).
enum GLErrorCode : uint8_t
{
    GL_DISABLED = 0,  // Idle
    GL_ENABLED = 1,   // Normal operation
    GL_OVER_VOLTAGE = 8,
    GL_UNDER_VOLTAGE = 9,
    GL_OVER_CURRENT = 10,
    GL_MOS_OVER_TEMP = 11,
    GL_MOTOR_OVER_TEMP = 12,
    GL_COMM_LOSS = 13,
    GL_OVERLOAD = 14,
};

// GL II config command payload suffixes (preceded by seven 0xFF bytes).
enum GLCommandSuffix : uint8_t
{
    GL_CMD_ENTER = 0xFC,       // Enter motor control mode (enable)
    GL_CMD_EXIT = 0xFD,        // Exit motor control mode (disable / free-run)
    GL_CMD_SET_ZERO = 0xFE,    // Set current position as zero
    GL_CMD_CLEAR_ERROR = 0xFB, // Clear diagnostic error states
};

// Self-contained CAN message for GL II. Standard 11-bit frames only — never touches
// CanMessage and never sets CAN_EFF_FLAG.
struct GLCanMessage : can_frame
{
    GLCanMessage();

    // MIT mode (mode 0) motion control. Inputs are already packed to scaled unsigned
    // ints (pos 16-bit, vel/kp/kd/torque 12-bit). CAN ID = node id, len = 8.
    void msgMotionControl(const uint8_t &node_id, const uint16_t &pos, const uint16_t &vel,
                          const uint16_t &kp, const uint16_t &kd, const uint16_t &torque);

    // Position-Velocity mode (mode 1). CAN ID = 0x100 | node_id. Payload = two IEEE-754
    // float32, little-endian: target position [rad] then velocity limit [rad/s].
    void msgMotionPosVel(const uint8_t &node_id, const float &pos, const float &vel);

    // Velocity mode (mode 2). CAN ID = 0x200 | node_id. Payload = one float32 [rad/s], LE.
    void msgMotionVel(const uint8_t &node_id, const float &vel);

    // Universal config command: payload FF FF FF FF FF FF FF <suffix>.
    // CAN ID = (mode << 8) | node_id. For MIT mode (0) this is just node_id.
    void msgCommand(const uint8_t &node_id, const uint8_t &mode, const uint8_t &suffix);

    bool send(const int &socket_fd, bool should_print = true); // returns true on success
};

class GLMotorController
{
public:
    int num_motors;

    std::vector<std::string> can_interfaces;
    Eigen::Matrix<uint8_t, Eigen::Dynamic, 1> can_interface_ids; // index of the CAN interface of the motor
    Eigen::Matrix<uint8_t, Eigen::Dynamic, 1> can_ids;           // node id of the motor
    Eigen::Matrix<uint8_t, Eigen::Dynamic, 1> motor_types;

    // Controller-wide control mode (GLControlMode). Determines both the config-command
    // CAN IDs and the motion frame format. All motors on a GL bus share one mode.
    uint8_t control_mode = GL_MODE_MIT;

    std::vector<MotorSpec> motor_specs;

    // Feedback frames all arrive on the Master ID (default 0); the responding motor is
    // identified by the low nibble of its node id in Data[0]&0x0F. Require node ids to be
    // distinct in their low nibble (1..15). -1 = unknown.
    std::array<int8_t, 16> gl_nibble_to_idx;

    // MIT command references (wr)
    Eigen::Matrix<float, Eigen::Dynamic, 1> mech_pos_ref;    // desired position [rad]
    Eigen::Matrix<float, Eigen::Dynamic, 1> mech_vel_ref;    // desired velocity [rad/s]
    Eigen::Matrix<float, Eigen::Dynamic, 1> mech_torque_ref; // feedforward torque [Nm]
    Eigen::Matrix<float, Eigen::Dynamic, 1> kp;              // MIT impedance position gain [N/rad]
    Eigen::Matrix<float, Eigen::Dynamic, 1> kd;              // MIT impedance velocity gain [Ns/rad]

    // Feedback state (ro)
    Eigen::Matrix<float, Eigen::Dynamic, 1> mech_pos;    // current position [rad]
    Eigen::Matrix<float, Eigen::Dynamic, 1> mech_vel;    // current velocity [rad/s]
    Eigen::Matrix<float, Eigen::Dynamic, 1> mech_torque; // current torque [Nm]
    Eigen::Matrix<float, Eigen::Dynamic, 1> drive_temp;  // driver board temperature [°C]
    Eigen::Matrix<float, Eigen::Dynamic, 1> motor_temp;  // motor winding temperature [°C]
    Eigen::Matrix<uint8_t, Eigen::Dynamic, 1> mode_status; // mapped: Reset/Running/Unknown
    Eigen::Matrix<uint8_t, Eigen::Dynamic, 1> error_code;  // raw GL err nibble (GLErrorCode)

    Eigen::Matrix<int, Eigen::Dynamic, 1> socket_fds;

    // Thread-safe queue for received CAN frames
    std::queue<can_frame> queue_recv;
    std::mutex mutex_recv;
    std::atomic<bool> receive_thread_should_run{true};
    std::vector<std::thread> receiver_threads;
    std::vector<std::thread> sender_threads;
    std::thread motion_control_thread;
    bool motion_control_thread_started = false;

    bool should_print_send = true;
    bool should_print_recv = true;

    LatencyLogger latency_logger;

    int max_try_send = 50;                // maximum number of tries to send the same CAN frame
    int response_wait_microsecond = 4000; // typical waiting time needed for motor to respond
    int min_same_bus_delay_us = 40;       // minimum delay between sends on same CAN bus (busy wait)
    int recv_timeout_ms = 2000;           // receiver poll timeout (see CanMotorController for rationale)

    // Delete copy; allow move
    GLMotorController(const GLMotorController &) = delete;
    GLMotorController &operator=(const GLMotorController &) = delete;
    GLMotorController(GLMotorController &&) noexcept = default;
    GLMotorController &operator=(GLMotorController &&) noexcept = default;

    GLMotorController(
        std::vector<std::string> can_interfaces,
        Eigen::Matrix<uint8_t, Eigen::Dynamic, 1> can_interface_ids,
        Eigen::Matrix<uint8_t, Eigen::Dynamic, 1> can_ids,
        Eigen::Matrix<uint8_t, Eigen::Dynamic, 1> motor_types,
        uint8_t control_mode = GL_MODE_MIT)
        : socket_fds(can_ids.size()),
          receiver_threads(can_interfaces.size()),
          sender_threads(can_interfaces.size()),
          mech_pos_ref(can_ids.size()),
          mech_vel_ref(can_ids.size()),
          mech_torque_ref(can_ids.size()),
          kp(can_ids.size()),
          kd(can_ids.size()),
          mech_pos(can_ids.size()),
          mech_vel(can_ids.size()),
          mech_torque(can_ids.size()),
          drive_temp(can_ids.size()),
          motor_temp(can_ids.size()),
          mode_status(can_ids.size()),
          error_code(can_ids.size()),
          latency_logger(can_ids.size(), 600)
    {
        this->can_interfaces = can_interfaces;
        this->can_interface_ids = can_interface_ids;
        this->can_ids = can_ids;
        this->num_motors = can_ids.size();
        this->motor_types = motor_types;
        this->control_mode = control_mode;

        // initialize references and status with zeros
        mech_pos_ref.setZero();
        mech_vel_ref.setZero();
        mech_torque_ref.setZero();
        kp.setZero();
        kd.setZero();
        mode_status.setZero();
        error_code.setZero();

        // Build lookup tables BEFORE spawning threads — parseFrame() reads these.
        gl_nibble_to_idx.fill(-1);
        motor_specs.resize(num_motors);
        for (int i = 0; i < num_motors; i++)
        {
            uint8_t nibble = can_ids(i) & 0x0F;
            if (gl_nibble_to_idx[nibble] != -1)
            {
                fmt::print(fmt_style_error(),
                           "[{0:%F %T}][{1}:{2}][{3}]: GL node ids must be distinct in low nibble; "
                           "0x{4:x} collides with motor #{5} (nibble 0x{6:x}).\033[0m\n",
                           get_time_now(), __FILE__, __LINE__, __func__, can_ids(i), gl_nibble_to_idx[nibble], nibble);
                throw std::runtime_error("GL node id low-nibble collision");
            }
            gl_nibble_to_idx[nibble] = (int8_t)i;
            motor_specs[i] = gl_spec_map[motor_types(i)];
        }

        build_round_robin_order();

        // Lock memory to prevent page faults (requires root/CAP_SYS_NICE)
        if (mlockall(MCL_CURRENT | MCL_FUTURE) != 0)
        {
            fmt::print(fmt_tyle_warn(), "[{0:%F %T}][{1}:{2}][{3}]: Failed to lock memory: {4}\033[0m\n",
                       get_time_now(), __FILE__, __LINE__, __func__, strerror(errno));
            fmt::print(fmt_tyle_warn(), "  → Running without RT memory locking (may have page fault latency spikes)\n");
            fmt::print(fmt_tyle_warn(), "  → To enable: run with sudo OR setcap cap_ipc_lock=+ep <program>\033[0m\n");
        }

        // Start CAN sockets and receiver threads
        for (size_t i = 0; i < can_interfaces.size(); ++i)
        {
            if (!setup_socket(can_interfaces[i], socket_fds(i)))
            {
                fmt::print(fmt_style_error(), "[{0:%F %T}][{1}:{2}][{3}]: Failed to setup socket for {4}\033[0m\n",
                           get_time_now(), __FILE__, __LINE__, __func__, can_interfaces[i]);
                throw std::runtime_error("Socket setup failed for " + can_interfaces[i]);
            }
            receiver_threads[i] = std::thread(&GLMotorController::receive_frame_thread, this, i);
        }

        // Initialize per-bus send contexts, staging buffers, and spawn sender threads
        bus_send_contexts.reserve(can_interfaces.size());
        staged_frames.resize(can_interfaces.size());
        for (size_t i = 0; i < can_interfaces.size(); ++i)
        {
            bus_send_contexts.push_back(std::make_unique<BusSendContext>());
            sender_threads[i] = std::thread(&GLMotorController::send_frame_thread, this, (int)i);
        }
    }

    ~GLMotorController()
    {
        receive_thread_should_run = false;

        // Join motion_control_thread FIRST (before closing sockets) to avoid
        // "Bad file descriptor" from sending on closed sockets.
        if (motion_control_thread_started)
        {
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
        for (auto &ctx : bus_send_contexts)
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
            if (receiver_threads[i].joinable())
            {
                receiver_threads[i].join();
                fmt::print(fmt_tyle_ok(), "[{0:%F %T}][{1}:{2}][{3}]: shutdown receiver thread #{4}.\033[0m\n", get_time_now(), __FILE__, __LINE__, __func__, i);
            }
            else
            {
                fmt::print(fmt_tyle_warn(), "[{0:%F %T}][{1}:{2}][{3}]: receiver thread #{4} not joinable.\033[0m\n", get_time_now(), __FILE__, __LINE__, __func__, i);
            }
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

    std::vector<float> get_motor_latency_history(int motor_id)
    {
        return latency_logger.get_history(motor_id);
    }

    int get_bus_send_queue_depth(int bus_id)
    {
        auto &ctx = *bus_send_contexts[bus_id];
        std::lock_guard<std::mutex> lock(ctx.mutex);
        return (int)ctx.queue.size();
    }

    // Parse a GL II feedback frame (standard 11-bit, on Master ID).
    void parseFrame(can_frame &frame)
    {
        // Dedicated bus, but guard anyway: ignore any stray extended (Robstride) frame.
        if (frame.can_id & CAN_EFF_FLAG)
        {
            fmt::print(stderr, fmt_tyle_warn(), "[{0:%F %T}][{1}:{2}][{3}]: ignoring extended frame 0x{4:x} on GL bus.\033[0m\n",
                       get_time_now(), __FILE__, __LINE__, __func__, frame.can_id);
            return;
        }
        if (frame.len < 8)
            return;

        uint8_t err = uint8_t(frame.data[0] >> 4);
        uint8_t nibble = uint8_t(frame.data[0] & 0x0F);
        int8_t id = gl_nibble_to_idx[nibble];
        if (id < 0)
        {
            fmt::print(stderr, fmt_style_error(), "[{0:%F %T}][{1}:{2}][{3}]: Unknown GL node nibble: 0x{4:x} (can_id=0x{5:x})\033[0m\n",
                       get_time_now(), __FILE__, __LINE__, __func__, nibble, frame.can_id);
            return;
        }

        // latency: feedback closes the round-trip started in motion_control_once()
        double now = std::chrono::duration<double>(std::chrono::steady_clock::now().time_since_epoch()).count();
        latency_logger.log_end(id, now);

        const auto &spec = motor_specs[id];
        uint16_t pos_int = (uint16_t)((frame.data[1] << 8) | frame.data[2]);
        uint16_t spd_int = (uint16_t)((frame.data[3] << 4) | (frame.data[4] >> 4));
        uint16_t t_int = (uint16_t)(((frame.data[4] & 0x0F) << 8) | frame.data[5]);

        mech_pos(id) = uint16_to_float(pos_int, spec.pos_min, spec.pos_max, 16);
        mech_vel(id) = uint16_to_float(spd_int, spec.vel_min, spec.vel_max, 12);
        mech_torque(id) = uint16_to_float(t_int, spec.torque_min, spec.torque_max, 12);
        drive_temp(id) = (float)(int8_t)frame.data[6];
        motor_temp(id) = (float)(int8_t)frame.data[7];

        error_code(id) = err;
        mode_status(id) = (err == GL_ENABLED) ? MotorModeStatus::Running
                          : (err == GL_DISABLED) ? MotorModeStatus::Reset
                                                  : MotorModeStatus::Unknown;

        if (err != GL_DISABLED && err != GL_ENABLED)
        {
            fmt::print(fmt_style_error(), "[{0:%F %T}][{1}:{2}][{3}]: motor#{4:02d} Error:{5}{6}{7}{8}{9}{10}{11}\033[0m\n",
                       get_time_now(), __FILE__, __LINE__, __func__, id,
                       (err == GL_OVER_VOLTAGE) ? "[OVER VOLTAGE]" : "",
                       (err == GL_UNDER_VOLTAGE) ? "[UNDER VOLTAGE]" : "",
                       (err == GL_OVER_CURRENT) ? "[OVER CURRENT]" : "",
                       (err == GL_MOS_OVER_TEMP) ? "[MOS OVER TEMP]" : "",
                       (err == GL_MOTOR_OVER_TEMP) ? "[MOTOR OVER TEMP]" : "",
                       (err == GL_COMM_LOSS) ? "[COMM LOSS]" : "",
                       (err == GL_OVERLOAD) ? "[OVERLOAD]" : "");
        }
        if (should_print_recv)
            fmt::print("[RECV] motor#{:02d}: nibble: 0x{:x} pos: {:.3f} rad, vel: {:.2f} rad/s, torque: {:.2f} Nm, drive_temp: {:.0f} °C, motor_temp: {:.0f} °C, err: {}, status: {}\n",
                       id, nibble, mech_pos(id), mech_vel(id), mech_torque(id), drive_temp(id), motor_temp(id), error_code(id), mode_status(id));
    }

    void start_motion_control_continuously()
    {
        motion_control_thread = std::thread([this]()
                                            {
            struct sched_param param;
            param.sched_priority = 90;
            if (sched_setscheduler(0, SCHED_FIFO, &param) != 0) {
                fmt::print(fmt_tyle_warn(), "[{0:%F %T}][{1}:{2}][{3}]: Failed to set RT priority for motion control: {4}\033[0m\n",
                           get_time_now(), __FILE__, __LINE__, __func__, strerror(errno));
                fmt::print(fmt_tyle_warn(), "  → Running without RT scheduling (no guaranteed timing)\n");
                fmt::print(fmt_tyle_warn(), "  → To enable: run with sudo OR setcap cap_sys_nice=+ep <program>\033[0m\n");
            }
            motion_control_continuously(); });
        motion_control_thread_started = true;
    }

    void stop()
    {
        receive_thread_should_run = false;
        fmt::print(fmt_tyle_ok(), "[{0:%F %T}][{1}:{2}][{3}]: stop() called, signaling threads to exit.\033[0m\n", get_time_now(), __FILE__, __LINE__, __func__);
    }

    void motion_control_continuously()
    {
        while (receive_thread_should_run)
        {
            motion_control_once();
            std::this_thread::sleep_for(std::chrono::microseconds(response_wait_microsecond));
        }
        receive_thread_should_run = false;
        fmt::print(fmt_tyle_ok(), "[{0:%F %T}][{1}:{2}]: motion_control_thread exit success.\033[0m\n", get_time_now(), __FILE__, __LINE__, __func__);
    }

    void motion_control_once()
    {
        double now = std::chrono::duration<double>(std::chrono::steady_clock::now().time_since_epoch()).count();

        // Phase 1: compute all frames and stage per bus — no mutex held
        for (auto &buf : staged_frames)
            buf.clear();
        for (int id : round_robin_motor_order)
        {
            const auto &spec = motor_specs[id];
            // NaN guard: float_to_uint clamps numeric out-of-range but NaN bypasses both
            // comparisons → UB in the cast. Warn loudly; fallbacks are semantically safe.
            auto chk = [&](float v, float fb, const char *nm)
            {
                if (__builtin_expect(std::isnan(v), 0))
                {
                    fmt::print(fmt_style_error(), "[SAFETY] motor#{:02d} NaN {} → {:.3f}\n", id, nm, fb);
                    return fb;
                }
                return v;
            };
            GLCanMessage frame;
            if (control_mode == GL_MODE_POS_VEL)
            {
                // Cascaded position loop with a velocity limit. mech_vel_ref is the speed
                // LIMIT here (not a target velocity); kp/kd/torque_ref are unused (driver
                // internal gains). Raw floats — no scaling.
                float pos = chk(mech_pos_ref(id), mech_pos(id), "pos");
                float vlim = chk(mech_vel_ref(id), 0.0f, "vel_limit");
                frame.msgMotionPosVel(can_ids(id), pos, vlim);
            }
            else if (control_mode == GL_MODE_VEL)
            {
                float vel = chk(mech_vel_ref(id), 0.0f, "vel");
                frame.msgMotionVel(can_ids(id), vel);
            }
            else // GL_MODE_MIT
            {
                uint16_t pos_u16 = (uint16_t)float_to_uint(chk(mech_pos_ref(id), mech_pos(id), "pos"), spec.pos_min, spec.pos_max, 16);
                uint16_t vel_u12 = (uint16_t)float_to_uint(chk(mech_vel_ref(id), 0.0f, "vel"), spec.vel_min, spec.vel_max, 12);
                uint16_t kp_u12 = (uint16_t)float_to_uint(chk(kp(id), 0.0f, "kp"), spec.kp_min, spec.kp_max, 12);
                uint16_t kd_u12 = (uint16_t)float_to_uint(chk(kd(id), 0.0f, "kd"), spec.kd_min, spec.kd_max, 12);
                uint16_t tor_u12 = (uint16_t)float_to_uint(chk(mech_torque_ref(id), 0.0f, "torq"), spec.torque_min, spec.torque_max, 12);
                frame.msgMotionControl(can_ids(id), pos_u16, vel_u12, kp_u12, kd_u12, tor_u12);
            }

            latency_logger.log_start(id, now);
            staged_frames[can_interface_ids(id)].push_back(std::move(frame));
        }

        // Phase 2: bulk-enqueue per bus with one lock + one notify each
        for (size_t bus = 0; bus < can_interfaces.size(); ++bus)
        {
            if (staged_frames[bus].empty())
                continue;
            {
                std::lock_guard<std::mutex> lock(bus_send_contexts[bus]->mutex);
                for (auto &f : staged_frames[bus])
                    bus_send_contexts[bus]->queue.push(std::move(f));
            }
            bus_send_contexts[bus]->cv.notify_one();
        }
    }

    // Enable motors where mask[id] is true; unmasked motors are left as-is.
    void enable(const Eigen::Matrix<bool, Eigen::Dynamic, 1> &motor_enable_mask)
    {
        assert(motor_enable_mask.size() == num_motors);
        MotorModeStatus target_status = MotorModeStatus::Running;
        for (int id = 0; id < num_motors; id++)
            if (motor_enable_mask(id))
                mode_status(id) = MotorModeStatus::Unknown;
        bool success;
        int try_count = -1;
        while (++try_count <= max_try_send)
        {
            success = true;
            for (int id = 0; id < num_motors; id++)
            {
                if (!motor_enable_mask(id))
                    continue;
                if (mode_status(id) != target_status)
                {
                    GLCanMessage frame;
                    frame.msgCommand(can_ids(id), control_mode, GL_CMD_ENTER);
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

    void enable() { enable(Eigen::Matrix<bool, Eigen::Dynamic, 1>::Constant(num_motors, true)); }

    // Disable all motors. clear_fault also sends a clear-errors command.
    void disable(bool clear_fault = false)
    {
        MotorModeStatus expected_value = MotorModeStatus::Reset;
        mode_status.setConstant(MotorModeStatus::Unknown);
        bool success;
        int try_count = -1;
        while (++try_count <= max_try_send)
        {
            success = true;
            for (int id = 0; id < num_motors; id++)
            {
                if (mode_status(id) != expected_value)
                {
                    GLCanMessage frame;
                    frame.msgCommand(can_ids(id), control_mode, GL_CMD_EXIT);
                    send(id, frame);
                    if (clear_fault)
                    {
                        GLCanMessage clr;
                        clr.msgCommand(can_ids(id), control_mode, GL_CMD_CLEAR_ERROR);
                        send(id, clr);
                    }
                    success = false;
                }
            }
            if (success)
                break;
            std::this_thread::sleep_for(std::chrono::microseconds(response_wait_microsecond));
        }
        fmt::print(success ? fmt_tyle_ok() : fmt_style_error(), "[{0:%F %T}][{1}:{2}][{3}]: motors disable {4} after {5:d} try\033[0m\n",
                   get_time_now(), __FILE__, __LINE__, __func__, success ? "success" : "failed", try_count);
        if (!success)
            for (int id = 0; id < num_motors; id++)
                fmt::print(expected_value == mode_status(id) ? fmt_tyle_ok() : fmt_style_error(), "  motor#{:02d}={}\n", id, mode_status(id));
    }

    // Set all motors' current position to zero.
    void set_pos_zero()
    {
        mech_pos.setConstant(0.02f);
        float tolerance = 0.002f;
        float expected_value = 0.0f;
        bool success;
        int try_count = -1;
        while (++try_count <= max_try_send)
        {
            success = true;
            for (int id = 0; id < num_motors; id++)
            {
                if (std::abs(mech_pos(id) - expected_value) > tolerance)
                {
                    GLCanMessage frame;
                    frame.msgCommand(can_ids(id), control_mode, GL_CMD_SET_ZERO);
                    send(id, frame);
                    success = false;
                }
            }
            if (success && try_count > 0)
                break;
            std::this_thread::sleep_for(std::chrono::microseconds(response_wait_microsecond));
        }
        fmt::print(success ? fmt_tyle_ok() : fmt_style_error(), "[{0:%F %T}][{1}:{2}][{3}]: motors set_pos_zero {4} after {5:d} try\033[0m\n",
                   get_time_now(), __FILE__, __LINE__, __func__, success ? "success" : "failed", try_count);
        if (!success)
            for (int id = 0; id < num_motors; id++)
                fmt::print(std::abs(mech_pos(id) - expected_value) <= tolerance ? fmt_tyle_ok() : fmt_style_error(), "  motor#{:02d}={:.3f}\n", id, mech_pos(id));
    }

    // Clear diagnostic error states on all motors.
    void clear_error()
    {
        for (int id : round_robin_motor_order)
        {
            GLCanMessage frame;
            frame.msgCommand(can_ids(id), control_mode, GL_CMD_CLEAR_ERROR);
            send(id, frame);
            std::this_thread::sleep_for(std::chrono::microseconds(response_wait_microsecond));
        }
        fmt::print(fmt_tyle_ok(), "[{0:%F %T}][{1}:{2}][{3}]: clear_error command sent to all motors\033[0m\n",
                   get_time_now(), __FILE__, __LINE__, __func__);
    }

private:
    // Per-bus send queue: decouples the control thread from CAN socket writes,
    // allowing sends on different buses to happen in parallel.
    struct BusSendContext
    {
        std::queue<GLCanMessage> queue;
        std::mutex mutex;
        std::condition_variable cv;
        double last_send_time = 0.0;
    };
    std::vector<std::unique_ptr<BusSendContext>> bus_send_contexts;

    // Staging buffers for motion_control_once(): one per bus, capacity reused each cycle.
    std::vector<std::vector<GLCanMessage>> staged_frames;

    std::vector<int> round_robin_motor_order;

    void build_round_robin_order()
    {
        std::map<uint8_t, std::vector<int>> iface_to_ids;
        for (int i = 0; i < num_motors; ++i)
            iface_to_ids[can_interface_ids(i)].push_back(i);

        size_t max_len = 0;
        for (const auto &[_, group] : iface_to_ids)
            max_len = std::max(max_len, group.size());

        for (size_t k = 0; k < max_len; ++k)
        {
            for (size_t iface = 0; iface < can_interfaces.size(); ++iface)
            {
                auto it = iface_to_ids.find(iface);
                if (it != iface_to_ids.end() && k < it->second.size())
                    round_robin_motor_order.push_back(it->second[k]);
            }
        }
    }

    // Enqueue frame to the per-bus sender thread (returns immediately, non-blocking)
    inline void send(int i, GLCanMessage &frame)
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
        struct sched_param param;
        param.sched_priority = 90;
        if (sched_setscheduler(0, SCHED_FIFO, &param) != 0)
        {
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
        int try_count = 0;
        while (receive_thread_should_run)
        {
            int ret = poll(&fds, 1, recv_timeout_ms);
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
                        parseFrame(frame); // parse immediately
                    }
                }
            }
            else if (receive_thread_should_run == false)
            {
                break;
            }
            else
            {
                try_count++;
                if (try_count >= max_try_recv)
                {
                    if (ret == 0)
                        fmt::print(fmt_tyle_warn(), "[{0:%F %T}][{1}:{2}][{3}]: Timeout waiting for CAN frame on {4}.\033[0m\n", get_time_now(), __FILE__, __LINE__, __func__, can_interface.c_str());
                    else
                        fmt::print(fmt_tyle_warn(), "[{0:%F %T}][{1}:{2}][{3}]: poll failed on {4}: {5}\033[0m\n", get_time_now(), __FILE__, __LINE__, __func__, can_interface.c_str(), strerror(errno));
                    break;
                }
            }
        }
        receive_thread_should_run = false;
        fmt::print(fmt_tyle_ok(), "[{0:%F %T}][{1}:{2}][{3}]: receiving thread on {4} exit success.\033[0m\n", get_time_now(), __FILE__, __LINE__, __func__, can_interface.c_str());
    }

    void send_frame_thread(int bus_id)
    {
        struct sched_param param;
        param.sched_priority = 90;
        if (sched_setscheduler(0, SCHED_FIFO, &param) != 0)
        {
            fmt::print(fmt_tyle_warn(), "[{0:%F %T}][{1}:{2}][{3}]: Failed to set RT priority for sender #{4}: {5}\033[0m\n",
                       get_time_now(), __FILE__, __LINE__, __func__, bus_id, strerror(errno));
        }

        auto &ctx = *bus_send_contexts[bus_id];

        while (true)
        {
            GLCanMessage msg;
            {
                std::unique_lock<std::mutex> lock(ctx.mutex);
                ctx.cv.wait(lock, [&]
                            { return !ctx.queue.empty() || !receive_thread_should_run; });
                if (ctx.queue.empty())
                    break; // shutdown with empty queue — done
                msg = std::move(ctx.queue.front());
                ctx.queue.pop();
            }

            // Enforce minimum inter-frame gap on this bus (busy wait for sub-ms precision)
            {
                auto now = std::chrono::steady_clock::now();
                double now_sec = std::chrono::duration<double>(now.time_since_epoch()).count();
                double elapsed_us = (now_sec - ctx.last_send_time) * 1e6;
                if (elapsed_us < min_same_bus_delay_us)
                {
                    auto wait_until = now + std::chrono::microseconds(
                                                static_cast<int>(min_same_bus_delay_us - elapsed_us));
                    while (std::chrono::steady_clock::now() < wait_until)
                        ; // busy wait
                }
            }

            // Send with retries on EAGAIN (kernel TX buffer full)
            constexpr int max_retries = 10;
            constexpr int retry_delay_us = 50;
            for (int retry = 0; retry < max_retries && receive_thread_should_run; retry++)
            {
                if (msg.send(socket_fds(bus_id), should_print_send))
                {
                    ctx.last_send_time = std::chrono::duration<double>(
                                             std::chrono::steady_clock::now().time_since_epoch())
                                             .count();
                    break;
                }
                std::this_thread::sleep_for(std::chrono::microseconds(retry_delay_us));
            }
        }
        fmt::print(fmt_tyle_ok(), "[{0:%F %T}][{1}:{2}][{3}]: sender thread #{4} exit success.\033[0m\n",
                   get_time_now(), __FILE__, __LINE__, __func__, bus_id);
    }
};

#endif // __GL_MOTOR_HPP__
