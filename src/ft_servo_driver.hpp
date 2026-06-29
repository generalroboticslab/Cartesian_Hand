#pragma once

#include <string>
#include <vector>
#include <array>
#include <thread>
#include <mutex>
#include <atomic>
#include <cstdint>
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
        std::vector<u8>  id_buf(ids.begin(), ids.end());
        std::vector<s16> pos_buf(pos.begin(), pos.end());
        std::vector<u16> spd_buf(n, speed);
        std::vector<u8>  acc_buf(n, acc);
        std::vector<u16> trq_buf(n, torque);
        std::lock_guard<std::mutex> lk(bus_mutex_);
        hlscl_.SyncWritePosEx(id_buf.data(), n, pos_buf.data(),
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

    int read_position(int id) {
        std::lock_guard<std::mutex> lk(bus_mutex_);
        return hlscl_.ReadPos(id);
    }

    int read_speed(int id) {
        std::lock_guard<std::mutex> lk(bus_mutex_);
        return hlscl_.ReadSpeed(id);
    }

    int read_load(int id) {
        std::lock_guard<std::mutex> lk(bus_mutex_);
        return hlscl_.ReadLoad(id);
    }

    // raw byte, unit = 0.1 V
    int get_voltage(int id) {
        std::lock_guard<std::mutex> lk(bus_mutex_);
        return hlscl_.ReadVoltage(id);
    }

    // degrees Celsius
    int get_temperature(int id) {
        std::lock_guard<std::mutex> lk(bus_mutex_);
        return hlscl_.ReadTemper(id);
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
        running_ = true;
        poll_thread_ = std::thread(&FtServoDriver::poll_loop, this);
    }

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

    void poll_loop() {
        // One sync-read TX covers all IDs; each ID replies with 4 bytes:
        //   [pos_l, pos_h, spd_l, spd_h] at HLSCL_PRESENT_POSITION_L (56).
        // syncReadBegin allocates RX buffer (IDN*(rxLen+6) bytes); syncReadEnd frees it.
        const int rx_bytes = 6;  // pos_l, pos_h, spd_l, spd_h, load_l, load_h
        const u8 n = static_cast<u8>(poll_ids_.size());
        {
            std::lock_guard<std::mutex> lk(bus_mutex_);
            hlscl_.syncReadBegin(n, rx_bytes, /*timeout_ms=*/5);
        }

        while (running_) {
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
                    // syncReadPacketRx resets index to 0; toWrod reads pos[0-1], spd[2-3], load[4-5]
                    pos_cache_[id]  = static_cast<int16_t>(hlscl_.syncReadRxPacketToWrod(15));
                    spd_cache_[id]  = static_cast<int16_t>(hlscl_.syncReadRxPacketToWrod(15));
                    load_cache_[id] = static_cast<int16_t>(hlscl_.syncReadRxPacketToWrod(15));
                }
            }
            usleep(poll_interval_us_);
        }

        {
            std::lock_guard<std::mutex> lk(bus_mutex_);
            hlscl_.syncReadEnd();
        }
    }
};
