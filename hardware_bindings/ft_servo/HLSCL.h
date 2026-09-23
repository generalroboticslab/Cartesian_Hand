/*
 * HLSCL.h
 * 飞特HLS系列串行舵机应用层程序
 * 日期: 2025.9.27
 * 作者: 
 */

#ifndef _HLSCL_H
#define _HLSCL_H

//内存表定义
//-------EPROM(只读)--------
#define HLSCL_MODEL_L 3
#define HLSCL_MODEL_H 4

//-------EPROM(读写)--------
#define HLSCL_ID 5
#define HLSCL_BAUD_RATE 6
#define HLSCL_SECOND_ID 7
#define HLSCL_MIN_ANGLE_LIMIT_L 9
#define HLSCL_MIN_ANGLE_LIMIT_H 10
#define HLSCL_MAX_ANGLE_LIMIT_L 11
#define HLSCL_MAX_ANGLE_LIMIT_H 12
#define HLSCL_CW_DEAD 26
#define HLSCL_CCW_DEAD 27
#define HLSCL_OFS_L 31
#define HLSCL_OFS_H 32
#define HLSCL_MODE 33

//-------SRAM(读写)--------
#define HLSCL_TORQUE_ENABLE 40
#define HLSCL_ACC 41
#define HLSCL_GOAL_POSITION_L 42
#define HLSCL_GOAL_POSITION_H 43
#define HLSCL_GOAL_TORQUE_L 44
#define HLSCL_GOAL_TORQUE_H 45
#define HLSCL_GOAL_SPEED_L 46
#define HLSCL_GOAL_SPEED_H 47
#define HLSCL_TORQUE_LIMIT_L 48
#define HLSCL_TORQUE_LIMIT_H 49
#define HLSCL_LOCK 55

//-------SRAM(只读)--------
#define HLSCL_PRESENT_POSITION_L 56
#define HLSCL_PRESENT_POSITION_H 57
#define HLSCL_PRESENT_SPEED_L 58
#define HLSCL_PRESENT_SPEED_H 59
#define HLSCL_PRESENT_LOAD_L 60
#define HLSCL_PRESENT_LOAD_H 61
#define HLSCL_PRESENT_VOLTAGE 62
#define HLSCL_PRESENT_TEMPERATURE 63
#define HLSCL_MOVING 66
#define HLSCL_PRESENT_CURRENT_L 69
#define HLSCL_PRESENT_CURRENT_H 70

#include "SCSerial.h"

class HLSCL : public SCSerial
{
public:
	HLSCL();
	HLSCL(u8 End);
	HLSCL(u8 End, u8 Level);
	int WritePosEx(u8 ID, s16 Position, u16 Speed, u8 ACC = 0, u16 Torque = 0);//普通写单个舵机位置指令
	int RegWritePosEx(u8 ID, s16 Position, u16 Speed, u8 ACC = 0, u16 Torque = 0);//异步写单个舵机位置指令(RegWriteAction生效)
	void SyncWritePosEx(u8 ID[], u8 IDN, s16 Position[], u16 Speed[], u8 ACC[], u16 Torque[]);//同步写多个舵机位置指令
	void SyncWriteSpe(u8 ID[], u8 IDN, s16 Speed[], u8 ACC[], u16 Torque[]);//同步写多个舵机速度指令
	int ServoMode(u8 ID);//Servo模式
	int WheelMode(u8 ID);//恒速模式
	int EleMode(u8 ID);//恒力模式
	int WriteSpe(u8 ID, s16 Speed, u8 ACC = 0, u16 Torque = 0);//恒速模式控制指令
	int WriteEle(u8 ID, s16 Torque);//恒力模式控制指令
	int EnableTorque(u8 ID, u8 Enable);//扭力控制指令
	int unLockEprom(u8 ID);//eprom解锁
	int LockEprom(u8 ID);//eprom加锁
	int CalibrationOfs(u8 ID);//中位校准
	int FeedBack(int ID);//反馈舵机信息
	int ReadPos(int ID);//读位置
	int ReadSpeed(int ID);//读速度
	int ReadLoad(int ID);//读输出至电机的电压百分比(0~1000)
	int ReadVoltage(int ID);//读电压
	int ReadTemper(int ID);//读温度
	int ReadMove(int ID);//读移动状态
	int ReadCurrent(int ID);//读电流
private:
	u8 Mem[HLSCL_PRESENT_CURRENT_H-HLSCL_PRESENT_POSITION_L+1];
};

#endif