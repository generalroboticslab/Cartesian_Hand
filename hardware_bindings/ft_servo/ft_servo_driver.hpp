#pragma once

#include <string>
#include <vector>
#include <array>
#include <thread>
#include <mutex>
#include <atomic>
#include <cstdint>
#include <optional>
#include <tuple>
#include <stdexcept>
#include <unistd.h>
#include "HLSCL.h"

/**
 * FtServoDriver — C++ wrapper around FeeeTech HLSCL SDK.
 *
 * Background poll thread issues one sync-read TX for all IDs per interval,
 * decodes pos+speed into cache. Python reads hit cache (no serial, no GIL hold).
 *
 * Mutex order: always bus_mutex_ before cache_mutex_.
 * Assumptions:
 *   - HLS3915 firmware supports INST_SYNC_READ (0x82).
 *   - HLSCL::End = 0 (little-endian, default ctor).
 *   - Single FtServoDriver instance per UART bus.
 */
class FtServoDriver {
public:
    FtServoDriver(const std::string& port, int baud = 1000000) {
        if (!hlscl_.begin(baud, port.c_str()))
            throw std::runtime_error("Failed to open servo port: " + port);
    }

    ~FtServoDriver() { stop_poll(); hlscl_.end(); }

    // ── Motion ────────────────────────────────────────────────────────────────

    bool set_position(int id, int pos, int speed = 0, int acc = 50, int torque = 500) {
        std::lock_guard<std::mutex> lk(bus_mutex_);
        return hlscl_.WritePosEx(id, pos, speed, acc, torque) > 0;
    }

    bool set_positions(const std::vector<int>& ids, const std::vector<int>& pos,
                       int speed = 0, int acc = 50, int torque = 500) {
        const int n = ids.size();
        return set_positions(ids, pos, std::vector<int>(n, speed),
                             std::vector<int>(n, acc), std::vector<int>(n, torque));
    }

    // Per-servo gains. SyncWritePosEx already takes Speed[]/ACC[]/Torque[]
    // arrays, because INST_SYNC_WRITE broadcasts one packet in which each
    // servo reads its own slice — differing gains are the native case, not an
    // extra. The scalar overload above is the special case, and costs the same
    // on the wire. Without this, a caller wanting one joint at a different
    // torque has to fall back to N unicast writes.
    bool set_positions(const std::vector<int>& ids, const std::vector<int>& pos,
                       const std::vector<int>& speed, const std::vector<int>& acc,
                       const std::vector<int>& torque) {
        const size_t n = ids.size();
        if (pos.size() != n || speed.size() != n || acc.size() != n || torque.size() != n)
            throw std::invalid_argument(
                "set_positions: ids, pos, speed, acc and torque must be the same length");
        std::vector<u8>  id_buf(ids.begin(), ids.end());
        std::vector<s16> pos_buf(pos.begin(), pos.end());
        std::vector<u16> spd_buf(speed.begin(), speed.end());
        std::vector<u8>  acc_buf(acc.begin(), acc.end());
        std::vector<u16> trq_buf(torque.begin(), torque.end());
        std::lock_guard<std::mutex> lk(bus_mutex_);
        hlscl_.SyncWritePosEx(id_buf.data(), static_cast<u8>(n), pos_buf.data(),
                              spd_buf.data(), acc_buf.data(), trq_buf.data());
        return true;
    }

    bool set_speed(int id, int speed, int acc = 50, int torque = 500) {
        std::lock_guard<std::mutex> lk(bus_mutex_);
        return hlscl_.WriteSpe(id, speed, acc, torque) > 0;
    }

    void set_speeds(const std::vector<int>& ids, const std::vector<int>& speeds,
                    int acc = 50, int torque = 500) {
        const int n = ids.size();
        std::vector<u8>  id_buf(ids.begin(), ids.end());
        std::vector<s16> spd_buf(speeds.begin(), speeds.end());
        std::vector<u8>  acc_buf(n, static_cast<u8>(acc));
        std::vector<u16> trq_buf(n, static_cast<u16>(torque));
        std::lock_guard<std::mutex> lk(bus_mutex_);
        hlscl_.SyncWriteSpe(id_buf.data(), n, spd_buf.data(), acc_buf.data(), trq_buf.data());
    }

    void enable_torque(int id, bool on) {
        std::lock_guard<std::mutex> lk(bus_mutex_);
        hlscl_.EnableTorque(id, on ? 1 : 0);
    }

    void enable_torques(const std::vector<int>& ids, bool on) {
        for (int id : ids) enable_torque(id, on);
    }

    // TORQUE_LIMIT (reg 48/49), 0-1000, scales the ceiling the position loop's
    // output is allowed to reach. This is NOT the `torque` argument of
    // `set_positions`: that one lands in GOAL_TORQUE (44/45) via WritePosEx,
    // which is the setpoint constant-force mode drives (see `WriteEle`) and
    // does not bound a position-mode move. Nothing else here writes 48/49, so
    // without this the ceiling stays at whatever the servo booted with.
    //
    // SRAM, not EEPROM: no unlock, no write-cycle wear, and it reverts to the
    // stored default on power cycle -- so a limit has to be re-applied per
    // session rather than being a one-time fix to the unit.
    bool set_torque_limit(int id, int limit) {
        std::lock_guard<std::mutex> lk(bus_mutex_);
        return hlscl_.writeWord(static_cast<u8>(id), HLSCL_TORQUE_LIMIT_L,
                                static_cast<u16>(limit)) > 0;
    }

    void set_torque_limits(const std::vector<int>& ids,
                           const std::vector<int>& limits) {
        if (limits.size() != ids.size())
            throw std::invalid_argument(
                "set_torque_limits: ids and limits must be the same length");
        for (size_t i = 0; i < ids.size(); ++i)
            set_torque_limit(ids[i], limits[i]);
    }

    // Reads back what 48/49 actually holds. The write above is a broadcast-free
    // unicast with a status reply, but a driver that only writes cannot tell a
    // limit that took from one the firmware ignored -- which is the whole
    // question this register was added to answer. -1 on no reply.
    int get_torque_limit(int id) {
        std::lock_guard<std::mutex> lk(bus_mutex_);
        return hlscl_.readWord(static_cast<u8>(id), HLSCL_TORQUE_LIMIT_L);
    }

    // mode: 0 = position, 1 = wheel/speed, 2 = torque
    void set_mode(int id, int mode) {
        if (mode == 0) {
            { std::lock_guard<std::mutex> lk(bus_mutex_); hlscl_.WriteSpe(id, 0, 0, 0); }
            usleep(50000);
        }
        std::lock_guard<std::mutex> lk(bus_mutex_);
        if (mode == 0)      hlscl_.ServoMode(id);
        else if (mode == 1) hlscl_.WheelMode(id);
        else if (mode == 2) hlscl_.EleMode(id);
    }

    void set_modes(const std::vector<int>& ids, int mode) {
        if (mode == 0) {
            {
                std::lock_guard<std::mutex> lk(bus_mutex_);
                for (int id : ids) hlscl_.WriteSpe(id, 0, 0, 0);
            }
            usleep(50000);
        }
        std::lock_guard<std::mutex> lk(bus_mutex_);
        for (int id : ids) {
            if      (mode == 0) hlscl_.ServoMode(id);
            else if (mode == 1) hlscl_.WheelMode(id);
            else if (mode == 2) hlscl_.EleMode(id);
        }
    }

    std::vector<int> scan(int start_id = 0, int end_id = 253) {
        std::vector<int> found;
        std::lock_guard<std::mutex> lk(bus_mutex_);
        for (int id = start_id; id <= end_id; ++id) {
            if (hlscl_.Ping(id) >= 0)
                found.push_back(id);
        }
        return found;
    }

    // ── Direct reads (one-shot: calibration, startup) ─────────────────────────
    //
    // These return nullopt (None in Python) when the read failed, rather than
    // returning the SDK's error value as if it were data.
    //
    // SCS::readWord returns -1 on any failure: no reply, wrong ID, bad length,
    // CRC mismatch. HLSCL::ReadPos then runs its sign-magnitude decode over
    // that -1 — bit 15 is set, so the "sign" branch fires and -1 comes back out
    // as +32769, a plausible-looking position roughly 400mm from zero. The
    // error sentinel and real data share one channel and are indistinguishable
    // downstream. getLastError() is the separate channel: Read() clears it on
    // success and sets it on every failure branch.

    std::optional<int> read_position(int id) {
        std::lock_guard<std::mutex> lk(bus_mutex_);
        int pos = hlscl_.ReadPos(id);
        if (hlscl_.getLastError()) return std::nullopt;
        return pos;
    }

    std::optional<int> read_speed(int id) {
        std::lock_guard<std::mutex> lk(bus_mutex_);
        int speed = hlscl_.ReadSpeed(id);
        if (hlscl_.getLastError()) return std::nullopt;
        return speed;
    }

    std::optional<int> read_load(int id) {
        std::lock_guard<std::mutex> lk(bus_mutex_);
        int load = hlscl_.ReadLoad(id);
        if (hlscl_.getLastError()) return std::nullopt;
        return load;
    }

    // raw byte, unit = 0.1 V
    std::optional<int> get_voltage(int id) {
        std::lock_guard<std::mutex> lk(bus_mutex_);
        int volt = hlscl_.ReadVoltage(id);
        if (hlscl_.getLastError()) return std::nullopt;
        return volt;
    }

    // degrees Celsius
    std::optional<int> get_temperature(int id) {
        std::lock_guard<std::mutex> lk(bus_mutex_);
        int temp = hlscl_.ReadTemper(id);
        if (hlscl_.getLastError()) return std::nullopt;
        return temp;
    }

    // One raw EPROM/SRAM byte. The register map in `HLSCL.h` stops at the
    // handful the driver needed and skips 13-25, which on this family hold the
    // protection settings -- temperature limit, voltage limits, max torque,
    // and the bitmask deciding which faults auto-clear TORQUE_ENABLE. Those
    // are exactly the registers that explain a low trip point, so reading them
    // beats guessing from the datasheet.
    std::optional<int> read_byte(int id, int addr) {
        std::lock_guard<std::mutex> lk(bus_mutex_);
        int value = hlscl_.readByte(id, addr);
        if (value == -1 || hlscl_.getLastError()) return std::nullopt;
        return value;
    }

    // milliamps, sign-magnitude like load: the sign is direction.
    //
    // This is the quantity the firmware's overload protection actually
    // integrates, so it is the one that says how close a stalled grip is to
    // having its TORQUE_ENABLE cleared out from under it. PRESENT_LOAD is a
    // duty-cycle proxy for the same thing and saturates at 1000; current does
    // not, so it keeps reporting once load has pinned.
    std::optional<int> get_current(int id) {
        std::lock_guard<std::mutex> lk(bus_mutex_);
        int current = hlscl_.ReadCurrent(id);
        if (hlscl_.getLastError()) return std::nullopt;
        return current;
    }

    // ── Vectorized read (one sync-read TX, one reply per servo) ───────────────

    // The read-side counterpart to set_positions: one broadcast request covers
    // every ID, against one request-plus-turnaround per servo for N calls to
    // read_position. A servo that does not answer yields nullopt rather than a
    // stale or fabricated value — sync-read reports a missing reply through
    // syncReadPacketRx's return, a separate channel from the data, so no error
    // sentinel is ever fed through the sign-magnitude decode.
    std::vector<std::optional<int>> read_positions(const std::vector<int>& ids) {
        if (running_)
            throw std::runtime_error(
                "read_positions: background poll is running and owns the "
                "sync-read buffer. Use get_positions(), or stop_poll() first.");

        const u8 n = static_cast<u8>(ids.size());
        std::vector<std::optional<int>> out(ids.size());
        if (n == 0) return out;

        const int rx_bytes = 6;   // pos_l, pos_h, spd_l, spd_h, load_l, load_h
        std::vector<u8> ids_u8(ids.begin(), ids.end());
        u8 pkt[6];

        std::lock_guard<std::mutex> lk(bus_mutex_);
        hlscl_.syncReadBegin(n, rx_bytes, /*timeout_ms=*/5);
        hlscl_.syncReadPacketTx(ids_u8.data(), n, HLSCL_PRESENT_POSITION_L, rx_bytes);
        for (size_t i = 0; i < ids.size(); ++i) {
            if (!hlscl_.syncReadPacketRx(ids_u8[i], pkt)) continue;   // stays nullopt
            // negBit=15: same sign-magnitude decode ReadPos applies.
            out[i] = hlscl_.syncReadRxPacketToWrod(15);
        }
        hlscl_.syncReadEnd();
        return out;
    }

    // Position, speed and load in the one sync-read that read_positions already
    // pays for. Registers 56-61 are contiguous -- position (56,57), speed
    // (58,59), load (60,61) -- so the 6-byte reply carries all three and
    // read_positions simply drops the last four bytes on the floor. Load
    // therefore costs no extra bus traffic, only the two extra decodes below.
    //
    // Separate from read_positions rather than replacing it: the control loop
    // wants positions only, and widening its return type would churn every
    // caller and every test for data that loop does not use.
    //
    // Sign-bit position is NOT uniform across these three registers. Position
    // and speed are 16-bit sign-magnitude (bit 15); Present Load is 11-bit with
    // its direction bit at 10 and magnitude 0-1000. Decoding load at 15 leaves
    // the direction bit inside the magnitude, so one push direction reads
    // 0-1000 and the other 1024-2024 -- a load channel that looks broken and
    // asymmetric rather than wrong. HLSCL::ReadLoad, the single-servo path, has
    // always used 10; these sync-read paths did not.
    static constexpr u8 NEG_BIT_POS_SPD = 15;
    static constexpr u8 NEG_BIT_LOAD    = 10;

    std::vector<std::optional<std::tuple<int, int, int>>>
    read_all(const std::vector<int>& ids) {
        if (running_)
            throw std::runtime_error(
                "read_all: background poll is running and owns the sync-read "
                "buffer. Use get_positions()/get_loads(), or stop_poll() first.");

        const u8 n = static_cast<u8>(ids.size());
        std::vector<std::optional<std::tuple<int, int, int>>> out(ids.size());
        if (n == 0) return out;

        const int rx_bytes = 6;   // pos_l, pos_h, spd_l, spd_h, load_l, load_h
        std::vector<u8> ids_u8(ids.begin(), ids.end());
        u8 pkt[6];

        std::lock_guard<std::mutex> lk(bus_mutex_);
        hlscl_.syncReadBegin(n, rx_bytes, /*timeout_ms=*/5);
        hlscl_.syncReadPacketTx(ids_u8.data(), n, HLSCL_PRESENT_POSITION_L, rx_bytes);
        for (size_t i = 0; i < ids.size(); ++i) {
            if (!hlscl_.syncReadPacketRx(ids_u8[i], pkt)) continue;   // stays nullopt
            // syncReadPacketRx resets the index to 0; each call advances one
            // word, so the order here IS the register order. All three are
            // sign-magnitude, but load's direction bit is at 10, not 15.
            const int pos  = hlscl_.syncReadRxPacketToWrod(NEG_BIT_POS_SPD);
            const int spd  = hlscl_.syncReadRxPacketToWrod(NEG_BIT_POS_SPD);
            const int load = hlscl_.syncReadRxPacketToWrod(NEG_BIT_LOAD);
            out[i] = std::make_tuple(pos, spd, load);
        }
        hlscl_.syncReadEnd();
        return out;
    }

    // returns ID on success, -1 on timeout
    int ping(int id) {
        std::lock_guard<std::mutex> lk(bus_mutex_);
        return hlscl_.Ping(id);
    }

    // unlock EPROM, write new ID, re-lock; returns true on success
    bool write_id(int id, int new_id) {
        std::lock_guard<std::mutex> lk(bus_mutex_);
        hlscl_.unLockEprom(static_cast<u8>(id));
        int ret = hlscl_.writeByte(static_cast<u8>(id), HLSCL_ID, static_cast<u8>(new_id));
        hlscl_.LockEprom(static_cast<u8>(new_id));
        return ret >= 0;
    }

    // ── Cached reads (real-time, no serial, no GIL hold) ─────────────────────

    int get_load(int id) const {
        std::lock_guard<std::mutex> lk(cache_mutex_);
        return load_cache_[id];
    }

    std::vector<int> get_loads(const std::vector<int>& ids) const {
        std::lock_guard<std::mutex> lk(cache_mutex_);
        std::vector<int> out;
        out.reserve(ids.size());
        for (int id : ids) out.push_back(load_cache_[id]);
        return out;
    }

    int get_position(int id) const {
        std::lock_guard<std::mutex> lk(cache_mutex_);
        return pos_cache_[id];
    }

    int get_speed(int id) const {
        std::lock_guard<std::mutex> lk(cache_mutex_);
        return spd_cache_[id];
    }

    std::vector<int> get_positions(const std::vector<int>& ids) const {
        std::lock_guard<std::mutex> lk(cache_mutex_);
        std::vector<int> out;
        out.reserve(ids.size());
        for (int id : ids) out.push_back(pos_cache_[id]);
        return out;
    }

    std::vector<int> get_speeds(const std::vector<int>& ids) const {
        std::lock_guard<std::mutex> lk(cache_mutex_);
        std::vector<int> out;
        out.reserve(ids.size());
        for (int id : ids) out.push_back(spd_cache_[id]);
        return out;
    }

    // ── Background poll ───────────────────────────────────────────────────────

    void start_poll(const std::vector<int>& ids, int interval_us = 5000) {
        stop_poll();
        poll_ids_ = ids;
        poll_interval_us_ = interval_us;
        poll_silence_ = 0;
        running_ = true;
        poll_thread_ = std::thread(&FtServoDriver::poll_loop, this);
    }

    // Consecutive poll sweeps in which no servo replied at all; 0 while healthy.
    int poll_silence() const { return poll_silence_; }

    void stop_poll() {
        running_ = false;
        if (poll_thread_.joinable()) poll_thread_.join();
    }

    void close() {
        stop_poll();
        hlscl_.end();
    }

private:
    HLSCL hlscl_;
    mutable std::mutex bus_mutex_;   // serializes all serial I/O
    mutable std::mutex cache_mutex_; // protects pos_cache_ / spd_cache_

    std::array<int16_t, 256> pos_cache_{};
    std::array<int16_t, 256> spd_cache_{};
    std::array<int16_t, 256> load_cache_{};
    std::vector<int> poll_ids_;
    int poll_interval_us_ = 5000;

    std::thread poll_thread_;
    std::atomic<bool> running_{false};
    std::atomic<int> poll_silence_{0};

    void poll_loop() {
        // One sync-read TX covers all IDs; each ID replies with 6 bytes:
        //   [pos_l, pos_h, spd_l, spd_h, load_l, load_h] from registers 56-61,
        //   which are contiguous starting at HLSCL_PRESENT_POSITION_L (56).
        // syncReadBegin allocates RX buffer (IDN*(rxLen+6) bytes); syncReadEnd frees it.
        const int rx_bytes = 6;  // pos_l, pos_h, spd_l, spd_h, load_l, load_h
        const u8 n = static_cast<u8>(poll_ids_.size());
        {
            std::lock_guard<std::mutex> lk(bus_mutex_);
            hlscl_.syncReadBegin(n, rx_bytes, /*timeout_ms=*/5);
        }

        while (running_) {
            int replied = 0;
            {
                std::lock_guard<std::mutex> lk(bus_mutex_);
                std::vector<u8> ids_u8(poll_ids_.begin(), poll_ids_.end());
                hlscl_.syncReadPacketTx(ids_u8.data(), n,
                                        HLSCL_PRESENT_POSITION_L, rx_bytes);

                // cache_mutex_ nested inside bus_mutex_ — consistent lock order
                std::lock_guard<std::mutex> ck(cache_mutex_);
                u8 pkt[6];
                for (int id : poll_ids_) {
                    if (!hlscl_.syncReadPacketRx(static_cast<u8>(id), pkt)) continue;
                    ++replied;
                    // syncReadPacketRx resets index to 0; toWrod reads pos[0-1], spd[2-3], load[4-5]
                    pos_cache_[id]  = static_cast<int16_t>(hlscl_.syncReadRxPacketToWrod(NEG_BIT_POS_SPD));
                    spd_cache_[id]  = static_cast<int16_t>(hlscl_.syncReadRxPacketToWrod(NEG_BIT_POS_SPD));
                    load_cache_[id] = static_cast<int16_t>(hlscl_.syncReadRxPacketToWrod(NEG_BIT_LOAD));
                }
            }
            // A sweep nobody answered means the bus is gone, not that one servo
            // glitched. Counted rather than acted on: this thread does not know
            // what the caller wants done, and get_* cannot report it -- a cache
            // the poll stopped refreshing reads exactly like a hand holding
            // still. Without this a caller polling a dead bus spins forever on
            // plausible stale numbers.
            if (replied == 0) ++poll_silence_;
            else poll_silence_ = 0;
            usleep(poll_interval_us_);
        }

        {
            std::lock_guard<std::mutex> lk(bus_mutex_);
            hlscl_.syncReadEnd();
        }
    }
};
