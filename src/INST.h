/*
 * INST.h
 * 飞特串行舵机协议指令及类型定义
 * Feetech SCS servo protocol — instruction codes, error codes, type aliases.
 */

#ifndef _INST_H
#define _INST_H

#include <stdint.h>

typedef uint8_t  u8;
typedef uint16_t u16;
typedef uint32_t u32;
typedef int16_t  s16;

// ── Instruction set ───────────────────────────────────────────────────────────
#define INST_PING        0x01
#define INST_READ        0x02
#define INST_WRITE       0x03
#define INST_REG_WRITE   0x04
#define INST_REG_ACTION  0x05
#define INST_RESET       0x06
#define INST_CAL         0x07
#define INST_SYNC_READ   0x82
#define INST_SYNC_WRITE  0x83

// ── Communication error codes ─────────────────────────────────────────────────
#define ERR_NO_REPLY     1
#define ERR_SLAVE_ID     2
#define ERR_BUFF_LEN     3
#define ERR_CRC_CMP      4

#endif
