/**
 * EasyObjectDictionary.cpp
 * @author COPYRIGHT(c) 2017 SYD Dynamics ApS
 * @see    EasyObjectDictionary.h for more descriptions.
 */
#include "EasyObjectDictionary.h"


EasyObjectDictionary::EasyObjectDictionary()
{	
    // Find the maximum size:
    int iDS=0, oDS=0;
	if( iDS < (int)sizeof(Ep_Request))       iDS = sizeof(Ep_Request);
    if( iDS < (int)sizeof(Ep_Ack))           iDS = sizeof(Ep_Ack);
    if( iDS < (int)sizeof(Ep_Status))        iDS = sizeof(Ep_Status);
    if( iDS < (int)sizeof(Ep_Raw_GyroAccMag))iDS = sizeof(Ep_Raw_GyroAccMag);
    if( iDS < (int)sizeof(Ep_Q_s1_s))        iDS = sizeof(Ep_Q_s1_s);
    if( iDS < (int)sizeof(Ep_Q_s1_e))        iDS = sizeof(Ep_Q_s1_e);
    if( iDS < (int)sizeof(Ep_Euler_s1_s))    iDS = sizeof(Ep_Euler_s1_s);
    if( iDS < (int)sizeof(Ep_Euler_s1_e))    iDS = sizeof(Ep_Euler_s1_e);
    if( iDS < (int)sizeof(Ep_RPY))           iDS = sizeof(Ep_RPY);
    if( iDS < (int)sizeof(Ep_Gravity))       iDS = sizeof(Ep_Gravity);
    if( iDS < (int)sizeof(Ep_Combo))         iDS = sizeof(Ep_Combo);
    oDS = iDS;
    maxSize = oDS;

    // QT Meta-Type Registration:
#   ifdef EP_PLATFORM_QT5_
    qRegisterMetaType<Ep_Header>("Ep_Header");
	qRegisterMetaType<Ep_Request>("Ep_Request");
    qRegisterMetaType<Ep_Ack>("Ep_Ack");
    qRegisterMetaType<Ep_Status>("Ep_Status");
    qRegisterMetaType<Ep_Raw_GyroAccMag>("Ep_Raw_GyroAccMag");
    qRegisterMetaType<Ep_Q_s1_s>("Ep_Q_s1_s");
    qRegisterMetaType<Ep_Q_s1_e>("Ep_Q_s1_e");
    qRegisterMetaType<Ep_Euler_s1_s>("Ep_Euler_s1_s");
    qRegisterMetaType<Ep_Euler_s1_e>("Ep_Euler_s1_e");
    qRegisterMetaType<Ep_RPY>("Ep_RPY");
    qRegisterMetaType<Ep_Gravity>("Ep_Gravity");
    qRegisterMetaType<Ep_Combo>("Ep_Combo");
#   endif

    // Value Reset:
    ep_Raw_GyroAccMag.timeStamp = 0;
    ep_Raw_GyroAccMag.acc[0] = 0;
    ep_Raw_GyroAccMag.acc[1] = 0;
    ep_Raw_GyroAccMag.acc[2] = 0;
    ep_Raw_GyroAccMag.gyro[0]= 0;
    ep_Raw_GyroAccMag.gyro[1]= 0;
    ep_Raw_GyroAccMag.gyro[2]= 0;
    ep_Raw_GyroAccMag.mag[0] = 0;
    ep_Raw_GyroAccMag.mag[1] = 0;
    ep_Raw_GyroAccMag.mag[2] = 0;
    
    // Initialize Dynamic Database of Object Items:
    {
        EOD_DB_Dynamic  arr[ EOD_DB_SIZE_ ] = EOD_DB_DYNAMIC_INIT;
        for(int i=0; i<((signed)(EOD_DB_SIZE_ * (sizeof(EOD_DB_Dynamic)))); i++){
            *(((char*)(eOD_DB_Dynamic))+i) = *(((char*)(arr))+i);
        }
    }
}


EasyObjectDictionary::~EasyObjectDictionary(){

}


int EasyObjectDictionary::Get_MaxSize(){
    return maxSize;
}






//------------------------------------------------------------------------
//                 Object Specific Read & Write Operations
//------------------------------------------------------------------------
/** @note Use the following method to read/write are intrinsically "safe" since they
  *       use the Read() and Write() method for low-level operation, which uses
  *       mutex to ensure the integrality of the data, and protect from unwanted changes
  *       of Object data once it is WriteProtected.
  *
  * @return EP_SUCC_            Means Operation done successfully
  *         EP_FAIL_            lengthIn doesn't match the type size, or the cmd does not exist in the object dictionary
  *         EP_MUTEX_LOCKED_    Permission denied while trying to access the Object.
 */

int EasyObjectDictionary::Write_Ep_Ack(
    EP_ID_TYPE_  toId,        ///< [INPUT] Destination ID of the package when it is to be sent
    EP_CMD_TYPE_ cmdAck       ///< [INPUT] The cmd it acknowledges to
){
    Ep_Ack          ep_Ack;
    Ep_Header    headerOut;
    ep_Ack.header.cmd    = EP_CMD_ACK_;
    ep_Ack.header.fromId = global_SysShortId  & EP_ID_MASK_;
    ep_Ack.header.qos    = global_SysQoS      & EP_QOS_MASK_;
    ep_Ack.header.toId   = toId               & EP_ID_MASK_;
    ep_Ack.cmdAck        = cmdAck;
    return Write((char*)(&ep_Ack), sizeof(Ep_Ack),  &headerOut);
}

int EasyObjectDictionary::Write_Ep_Status(
    EP_ID_TYPE_  toId,        ///< [INPUT]
    uint32  timeStamp,        ///< [INPUT]
    float32 temperature,      ///< [INPUT]
    uint16  updateRate        ///< [INPUT]
){
    Ep_Status    ep_Status;
    Ep_Header    headerOut;
    ep_Status.header.cmd    = EP_CMD_STATUS_;
    ep_Status.header.fromId = global_SysShortId  & EP_ID_MASK_;
    ep_Status.header.qos    = global_SysQoS      & EP_QOS_MASK_;
    ep_Status.header.toId   = toId               & EP_ID_MASK_;
    ep_Status.timeStamp  = timeStamp;
    ep_Status.temperature= temperature;
    ep_Status.updateRate = updateRate;
    return Write((char*)(&ep_Status), sizeof(Ep_Status),  &headerOut);
}

int EasyObjectDictionary::Write_Ep_Q_s1_s(
    EP_ID_TYPE_  toId,                               ///< [INPUT]
    uint32 timeStamp,                                ///< [INPUT]
    float q1, float q2, float q3, float q4           ///< [INPUT]
){
    Ep_Q_s1_s        ep_Q_s1_s;
    Ep_Header        headerOut;
    ep_Q_s1_s.header.cmd  = EP_CMD_Q_S1_S_;
    ep_Q_s1_s.header.fromId = global_SysShortId  & EP_ID_MASK_;
    ep_Q_s1_s.header.qos    = global_SysQoS      & EP_QOS_MASK_;
    ep_Q_s1_s.header.toId   = toId               & EP_ID_MASK_;
    ep_Q_s1_s.timeStamp   = timeStamp;
    ep_Q_s1_s.q[0] = q1;
    ep_Q_s1_s.q[1] = q2;
    ep_Q_s1_s.q[2] = q3;
    ep_Q_s1_s.q[3] = q4;
    return Write((char*)(&ep_Q_s1_s), sizeof(Ep_Q_s1_s),  &headerOut);
}

int EasyObjectDictionary::Write_Ep_Raw_GyroAccMag(
    EP_ID_TYPE_  toId,                             ///< [INPUT]
    uint32 timeStamp,                              ///< [INPUT]
    float wx, float wy, float wz,                  ///< [INPUT]
    float ax, float ay, float az,                  ///< [INPUT]
    float mx, float my, float mz                   ///< [INPUT]
){
    Ep_Raw_GyroAccMag    ep_Raw_GyroAccMag;
    Ep_Header            headerOut;
    ep_Raw_GyroAccMag.header.cmd    = EP_CMD_Raw_GYRO_ACC_MAG_;
    ep_Raw_GyroAccMag.header.fromId = global_SysShortId  & EP_ID_MASK_;
    ep_Raw_GyroAccMag.header.qos    = global_SysQoS      & EP_QOS_MASK_;
    ep_Raw_GyroAccMag.header.toId   = toId               & EP_ID_MASK_;
    ep_Raw_GyroAccMag.timeStamp     = timeStamp;
    ep_Raw_GyroAccMag.gyro[0] = wx; 
    ep_Raw_GyroAccMag.gyro[1] = wy; 
    ep_Raw_GyroAccMag.gyro[2] = wz;
    ep_Raw_GyroAccMag.acc[0]  = ax;
    ep_Raw_GyroAccMag.acc[1]  = ay;
    ep_Raw_GyroAccMag.acc[2]  = az;
    ep_Raw_GyroAccMag.mag[0]  = mx;
    ep_Raw_GyroAccMag.mag[1]  = my;
    ep_Raw_GyroAccMag.mag[2]  = mz;
    return Write((char*)(&ep_Raw_GyroAccMag), sizeof(Ep_Raw_GyroAccMag), &headerOut);
}

int EasyObjectDictionary::Write_Ep_Euler_s1_s(
    EP_ID_TYPE_  toId,                             ///< [INPUT]
    uint32 timeStamp,                              ///< [INPUT]
    float psi, float theta, float phi              ///< [INPUT]
){
    Ep_Euler_s1_s    ep_Euler_s1_s;
    Ep_Header        headerOut;
    ep_Euler_s1_s.header.cmd    = EP_CMD_EULER_S1_S_;
    ep_Euler_s1_s.header.fromId = global_SysShortId  & EP_ID_MASK_;
    ep_Euler_s1_s.header.qos    = global_SysQoS      & EP_QOS_MASK_;
    ep_Euler_s1_s.header.toId   = toId               & EP_ID_MASK_;
    ep_Euler_s1_s.timeStamp     = timeStamp;
    ep_Euler_s1_s.psi   = psi;
    ep_Euler_s1_s.theta = theta;
    ep_Euler_s1_s.phi   = phi;
    return Write((char*)(&ep_Euler_s1_s), sizeof(Ep_Euler_s1_s),  &headerOut);
}

int EasyObjectDictionary::Write_Ep_Q_s1_e(
    EP_ID_TYPE_  toId,                             ///< [INPUT]
    uint32 timeStamp,                              ///< [INPUT]
    float q1, float q2, float q3, float q4         ///< [INPUT]
){
    Ep_Q_s1_e        ep_Q_s1_e;
    Ep_Header        headerOut;
    ep_Q_s1_e.header.cmd    = EP_CMD_Q_S1_E_;
    ep_Q_s1_e.header.fromId = global_SysShortId  & EP_ID_MASK_;
    ep_Q_s1_e.header.qos    = global_SysQoS      & EP_QOS_MASK_;
    ep_Q_s1_e.header.toId   = toId               & EP_ID_MASK_;
    ep_Q_s1_e.timeStamp     = timeStamp;
    ep_Q_s1_e.q[0] = q1;
    ep_Q_s1_e.q[1] = q2;
    ep_Q_s1_e.q[2] = q3;
    ep_Q_s1_e.q[3] = q4;
    return Write((char*)(&ep_Q_s1_e), sizeof(Ep_Q_s1_e),  &headerOut);
}

int EasyObjectDictionary::Write_Ep_Euler_s1_e(
    EP_ID_TYPE_  toId,                             ///< [INPUT]
    uint32 timeStamp,                              ///< [INPUT]
    float psi, float theta, float phi              ///< [INPUT]
){
    Ep_Euler_s1_e    ep_Euler_s1_e;
    Ep_Header        headerOut;
    ep_Euler_s1_e.header.cmd    = EP_CMD_EULER_S1_E_;
    ep_Euler_s1_e.header.fromId = global_SysShortId  & EP_ID_MASK_;
    ep_Euler_s1_e.header.qos    = global_SysQoS      & EP_QOS_MASK_;
    ep_Euler_s1_e.header.toId   = toId               & EP_ID_MASK_;
    ep_Euler_s1_e.timeStamp     = timeStamp;
    ep_Euler_s1_e.psi   = psi;
    ep_Euler_s1_e.theta = theta;
    ep_Euler_s1_e.phi   = phi;
    return Write((char*)(&ep_Euler_s1_e), sizeof(Ep_Euler_s1_e),  &headerOut);
}

int EasyObjectDictionary::Write_Ep_RPY(
    EP_ID_TYPE_  toId,                             ///< [INPUT]
    uint32 timeStamp,                              ///< [INPUT]
    float roll, float pitch, float yaw             ///< [INPUT]
){
    Ep_RPY           ep_RPY;
    Ep_Header        headerOut;
    ep_RPY.header.cmd    = EP_CMD_RPY_;
    ep_RPY.header.fromId = global_SysShortId  & EP_ID_MASK_;
    ep_RPY.header.qos    = global_SysQoS      & EP_QOS_MASK_;
    ep_RPY.header.toId   = toId               & EP_ID_MASK_;
    ep_RPY.timeStamp     = timeStamp;
    ep_RPY.roll    = roll;
    ep_RPY.pitch   = pitch;
    ep_RPY.yaw     = yaw;
    return Write((char*)(&ep_RPY), sizeof(Ep_RPY),  &headerOut); 
}

int EasyObjectDictionary::Write_Ep_Gravity(
    EP_ID_TYPE_  toId,                             ///< [INPUT]
    uint32 timeStamp,                              ///< [INPUT]
    float gravityX, float gravityY, float gravityZ ///< [INPUT]
){
    Ep_Gravity       ep_Gravity;
    Ep_Header        headerOut;
    ep_Gravity.header.cmd    = EP_CMD_GRAVITY_;
    ep_Gravity.header.fromId = global_SysShortId  & EP_ID_MASK_;
    ep_Gravity.header.qos    = global_SysQoS      & EP_QOS_MASK_;
    ep_Gravity.header.toId   = toId               & EP_ID_MASK_;
    ep_Gravity.timeStamp     = timeStamp;
    ep_Gravity.g[0] = gravityX;
    ep_Gravity.g[1] = gravityY;
    ep_Gravity.g[2] = gravityZ;
    return Write((char*)(&ep_Gravity), sizeof(Ep_Gravity),  &headerOut);
}

int EasyObjectDictionary::Write_Ep_Combo(
    EP_ID_TYPE_  toId,                             ///< [INPUT]
    uint32 timeStamp,                              ///< [INPUT]
    Ep_Status_SysState sys,                        ///< [INPUT]
    int16 roll,int16 pitch, uint16 yaw,            ///< [INPUT]
    int32 q1, int32 q2, int32 q3, int32 q4,        ///< [INPUT]
    int32 wx, int32 wy, int32 wz,                  ///< [INPUT]
    int32 ax, int32 ay, int32 az,                  ///< [INPUT]
    int16 mx, int16 my, int16 mz,                  ///< [INPUT]
    int8 temperature,uint8 updateRate,uint16 reserved1,uint16 simpleChecksum ///< [INPUT]
){
    Ep_Combo         ep_Combo;
    Ep_Header        headerOut;
    ep_Combo.header.cmd    = EP_CMD_COMBO_;
    ep_Combo.header.fromId = global_SysShortId  & EP_ID_MASK_;
    ep_Combo.header.qos    = global_SysQoS      & EP_QOS_MASK_;
    ep_Combo.header.toId   = toId               & EP_ID_MASK_;
    ep_Combo.timeStamp     = timeStamp;

    ep_Combo.sysState = sys;
    ep_Combo.roll = roll;ep_Combo.pitch = pitch;ep_Combo.yaw = yaw;
    ep_Combo.q1 = q1;ep_Combo.q2 = q2;ep_Combo.q3 = q3;ep_Combo.q4 = q4;
    ep_Combo.wx = wx;ep_Combo.wy = wy;ep_Combo.wz = wz;
    ep_Combo.ax = ax;ep_Combo.ay = ay;ep_Combo.az = az;
    ep_Combo.mx = mx;ep_Combo.my = my;ep_Combo.mz = mz;
    ep_Combo.temperature = temperature;
    ep_Combo.updateRate = updateRate;
    ep_Combo.reserved1 = reserved1;
    ep_Combo.simpleChecksum = simpleChecksum;

    return Write((char*)(&ep_Combo), sizeof(Ep_Combo),  &headerOut);
}

/**
 * @section Write_Ep_*() Example
   @code
        if(EP_SUCC_ == Write_Ep_Request( 123, EP_CMD_RPY_ )){
            // write operation successfuly done, do something ...
        }
   @endcode
 */
int EasyObjectDictionary::Write_Ep_Request(
    EP_ID_TYPE_       toId,                       ///< [INPUT]
    EP_CMD_TYPE_      cmdRequest                  ///< [INPUT]
){
    Ep_Request        ep_Request;
    Ep_Header         headerOut;
    ep_Request.header.cmd    = EP_CMD_REQUEST_;
    ep_Request.header.fromId = global_SysShortId  & EP_ID_MASK_;
    ep_Request.header.qos    = global_SysQoS      & EP_QOS_MASK_;
    ep_Request.header.toId   = toId               & EP_ID_MASK_;
    ep_Request.cmdRequest    = cmdRequest;
    return Write((char*)(&ep_Request), sizeof(Ep_Request),  &headerOut);
}


/**
 * @section Read_Ep_*() Example
 * @code
          Ep_Request ep_Request;
          if(EP_SUCC_ == eOD.Read_Ep_Request(&ep_Request)){
                // Use the data saved in ep_Request
          }
 * @endcode
 */
int EasyObjectDictionary::Read_Ep_Request(
    Ep_Request* dataOut
){
    EP_CMD_TYPE_  cmd = EP_CMD_REQUEST_;
    int           odLengthOut;
    char*         odDataOut;
    int           retVal = EP_FAIL_;
    if(dataOut == 0) return EP_FAIL_;
    
    retVal = Read(cmd,  &odDataOut, &odLengthOut);
    if(retVal == EP_SUCC_){
        if(odLengthOut == sizeof(Ep_Request)){
            for(int i=0; i<odLengthOut; i++){
              *(((char*)(dataOut))+i) = odDataOut[i];
            }
        }
    }
    EOD_DB_SetReadProtect(cmd, false);
    return retVal;
}

int EasyObjectDictionary::Read_Ep_Ack(
    Ep_Ack* dataOut
){
    EP_CMD_TYPE_  cmd = EP_CMD_ACK_;
    int           odLengthOut;
    char*         odDataOut;
    int           retVal = EP_FAIL_;
    if(dataOut == 0) return EP_FAIL_;

    retVal = Read(cmd,  &odDataOut, &odLengthOut);
    if(retVal == EP_SUCC_){
        if(odLengthOut == sizeof(Ep_Ack)){
            for(int i=0; i<odLengthOut; i++){
              *(((char*)(dataOut))+i) = odDataOut[i];
            }
        }
    }
    EOD_DB_SetReadProtect(cmd, false);
    return retVal;
}

int EasyObjectDictionary::Read_Ep_Status(
    Ep_Status* dataOut
){
    EP_CMD_TYPE_  cmd = EP_CMD_STATUS_;
    int           odLengthOut;
    char*         odDataOut;
    int           retVal = EP_FAIL_;
    if(dataOut == 0) return EP_FAIL_;

    retVal = Read(cmd,  &odDataOut, &odLengthOut);
    if(retVal == EP_SUCC_){
        if(odLengthOut == sizeof(Ep_Status)){
            for(int i=0; i<odLengthOut; i++){
              *(((char*)(dataOut))+i) = odDataOut[i];
            }
        }
    }
    EOD_DB_SetReadProtect(cmd, false);
    return retVal;
}

int EasyObjectDictionary::Read_Ep_Raw_GyroAccMag(
    Ep_Raw_GyroAccMag* dataOut
){
    EP_CMD_TYPE_  cmd = EP_CMD_Raw_GYRO_ACC_MAG_;
    int           odLengthOut;
    char*         odDataOut;
    int           retVal = EP_FAIL_;
    if(dataOut == 0) return EP_FAIL_;

    retVal = Read(cmd,  &odDataOut, &odLengthOut);
    if(retVal == EP_SUCC_){
        if(odLengthOut == sizeof(Ep_Raw_GyroAccMag)){
            for(int i=0; i<odLengthOut; i++){
              *(((char*)(dataOut))+i) = odDataOut[i];
            }
        }
    }
    EOD_DB_SetReadProtect(cmd, false);
    return retVal;
}

int EasyObjectDictionary::Read_Ep_Q_s1_s(
    Ep_Q_s1_s* dataOut
){
    EP_CMD_TYPE_  cmd = EP_CMD_Q_S1_S_;
    int           odLengthOut;
    char*         odDataOut;
    int           retVal = EP_FAIL_;
    if(dataOut == 0) return EP_FAIL_;

    retVal = Read(cmd,  &odDataOut, &odLengthOut);
    if(retVal == EP_SUCC_){
        if(odLengthOut == sizeof(Ep_Q_s1_s)){
            for(int i=0; i<odLengthOut; i++){
              *(((char*)(dataOut))+i) = odDataOut[i];
            }
        }
    }
    EOD_DB_SetReadProtect(cmd, false);
    return retVal;
}

int EasyObjectDictionary::Read_Ep_Euler_s1_s(
    Ep_Euler_s1_s* dataOut
){
    EP_CMD_TYPE_  cmd = EP_CMD_EULER_S1_S_;
    int           odLengthOut;
    char*         odDataOut;
    int           retVal = EP_FAIL_;
    if(dataOut == 0) return EP_FAIL_;

    retVal = Read(cmd,  &odDataOut, &odLengthOut);
    if(retVal == EP_SUCC_){
        if(odLengthOut == sizeof(Ep_Euler_s1_s)){
            for(int i=0; i<odLengthOut; i++){
              *(((char*)(dataOut))+i) = odDataOut[i];
            }
        }
    }
    EOD_DB_SetReadProtect(cmd, false);
    return retVal;
}

int EasyObjectDictionary::Read_Ep_Q_s1_e(
    Ep_Q_s1_e* dataOut
){
    EP_CMD_TYPE_  cmd = EP_CMD_Q_S1_E_;
    int           odLengthOut;
    char*         odDataOut;
    int           retVal = EP_FAIL_;
    if(dataOut == 0) return EP_FAIL_;

    retVal = Read(cmd,  &odDataOut, &odLengthOut);
    if(retVal == EP_SUCC_){
        if(odLengthOut == sizeof(Ep_Q_s1_e)){
            for(int i=0; i<odLengthOut; i++){
              *(((char*)(dataOut))+i) = odDataOut[i];
            }
        }
    }
    EOD_DB_SetReadProtect(cmd, false);
    return retVal;
}

int EasyObjectDictionary::Read_Ep_Euler_s1_e(
    Ep_Euler_s1_e* dataOut
){
    EP_CMD_TYPE_  cmd = EP_CMD_EULER_S1_E_;
    int           odLengthOut;
    char*         odDataOut;
    int           retVal = EP_FAIL_;
    if(dataOut == 0) return EP_FAIL_;

    retVal = Read(cmd,  &odDataOut, &odLengthOut);
    if(retVal == EP_SUCC_){
        if(odLengthOut == sizeof(Ep_Euler_s1_e)){
            for(int i=0; i<odLengthOut; i++){
              *(((char*)(dataOut))+i) = odDataOut[i];
            }
        }
    }
    EOD_DB_SetReadProtect(cmd, false);
    return retVal;
}

int EasyObjectDictionary::Read_Ep_RPY(
    Ep_RPY* dataOut
){
    EP_CMD_TYPE_  cmd = EP_CMD_RPY_;
    int           odLengthOut;
    char*         odDataOut;
    int           retVal = EP_FAIL_;
    if(dataOut == 0) return EP_FAIL_;

    retVal = Read(cmd,  &odDataOut, &odLengthOut);
    if(retVal == EP_SUCC_){
        if(odLengthOut == sizeof(Ep_RPY)){
            for(int i=0; i<odLengthOut; i++){
              *(((char*)(dataOut))+i) = odDataOut[i];
            }
        }
    }
    EOD_DB_SetReadProtect(cmd, false);
    return retVal;
}

int EasyObjectDictionary::Read_Ep_Gravity(
    Ep_Gravity* dataOut
){
    EP_CMD_TYPE_  cmd = EP_CMD_GRAVITY_;
    int           odLengthOut;
    char*         odDataOut;
    int           retVal = EP_FAIL_;
    if(dataOut == 0) return EP_FAIL_;

    retVal = Read(cmd,  &odDataOut, &odLengthOut);
    if(retVal == EP_SUCC_){
        if(odLengthOut == sizeof(Ep_Gravity)){
            for(int i=0; i<odLengthOut; i++){
              *(((char*)(dataOut))+i) = odDataOut[i];
            }
        }
    }
    EOD_DB_SetReadProtect(cmd, false);
    return retVal;
}


int EasyObjectDictionary::Read_Ep_Combo(
    Ep_Combo* dataOut
){
    EP_CMD_TYPE_  cmd = EP_CMD_COMBO_;
    int           odLengthOut;
    char*         odDataOut;
    int           retVal = EP_FAIL_;
    if(dataOut == 0) return EP_FAIL_;

    retVal = Read(cmd,  &odDataOut, &odLengthOut);
    if(retVal == EP_SUCC_){
        if(odLengthOut == sizeof(Ep_Combo)){
            for(int i=0; i<odLengthOut; i++){
              *(((char*)(dataOut))+i) = odDataOut[i];
            }
        }
    }
    EOD_DB_SetReadProtect(cmd, false);
    return retVal;
}
//------------------------------------------------------------------------
//              Object Specific Read & Write Operations
//------------------------------------------------------------------------






/**
 * EOD_DB_FindKey
 *
 * @return key of the item in DataBase.
 *          Range [0, (EOD_DB_SIZE_-1)]
 *          return -1 means not found
 */
int  EasyObjectDictionary::EOD_DB_FindKey(
    EP_CMD_TYPE_  cmdIn   ///< [INPUT] the search keyword cmd
){
    for(int i=0; i<EOD_DB_SIZE_; i++){
        if(eOD_DB_Static[i].cmd == cmdIn){
            return i;
        }
    }
    return -1;
}


/**
  * Set & Reset write protection flag
  * @note   When WriteProtect is enabled to a specific Object:
  *            (1) Write() permission denied: no changes can be made to the Object;
  *            (2) Read() permission is still avilable.
  * @return EP_SUCC_            Setup Successful
  *         EP_MUTEX_LOCKED_    The cmd is already write protected
  *         EP_FAIL_            cmd does not exist in the object dictionary,
  */
int  EasyObjectDictionary::EOD_DB_SetWriteProtect(
    EP_CMD_TYPE_  cmdIn,   ///< [INPUT] the search keyword cmd
    bool          enable   ///< [INPUT] true: enable write protection; false: disable.
){                                        
    int key = EOD_DB_FindKey(cmdIn);
    if( key == -1) {
        return EP_FAIL_;
    }
    else{
        if(enable){
            if((eOD_DB_Dynamic[key].mutex) & EOD_MUTEX_WRITE_PROTECT_)
                return EP_MUTEX_LOCKED_;
            // Mutex: set EOD_MUTEX_WRITE_PROTECT_ bit:
            eOD_DB_Dynamic[key].mutex |= EOD_MUTEX_WRITE_PROTECT_;
        }   
        else{
            // Mutex: clear EOD_MUTEX_WRITE_PROTECT_ bit:
            eOD_DB_Dynamic[key].mutex &= (~EOD_MUTEX_WRITE_PROTECT_); 
        }
    }
    return EP_SUCC_;
}


/**
  * Set & Reset read protection flag
  * @note   When ReadProtect is enabled to a specific Object:
  *            (1) Write() permission is denied: no changes can be made to the Object;
  *            (2) Read()  permission is also denied (to avoid another Read() from initiating
  *                           during the protection period, since the end of protection of the
  *                           previous Read() also means the end of protection of the latter,
  *                           which may lead to unexpected system behavior). 
  * @return EP_SUCC_            Setup Successful
  *         EP_MUTEX_LOCKED_    The cmd is already read protected
  *         EP_FAIL_            cmd does not exist in the object dictionary
  */
int  EasyObjectDictionary::EOD_DB_SetReadProtect(
    EP_CMD_TYPE_  cmdIn,   ///< [INPUT] the search keyword cmd
    bool          enable   ///< [INPUT] true: enable read protection; false: disable.
){                                        
    int key = EOD_DB_FindKey(cmdIn);
    if( key == -1) {
        return EP_FAIL_;
    }
    else{
        if(enable){
            if((eOD_DB_Dynamic[key].mutex) & EOD_MUTEX_READ_PROTECT_)
                return EP_MUTEX_LOCKED_;

            // Mutex: set EOD_MUTEX_READ_PROTECT_ bit :
            eOD_DB_Dynamic[key].mutex |= EOD_MUTEX_READ_PROTECT_;
        }   
        else{
            // Mutex: clear EOD_MUTEX_READ_PROTECT_ bit:
            eOD_DB_Dynamic[key].mutex &= (~EOD_MUTEX_READ_PROTECT_); 
        }
    }
    return EP_SUCC_;
}





//------------------------------------------------------------------------
//                                 Read
//------------------------------------------------------------------------
/**
  * Fetch the pointer to the Object from the dictinoary specificed the identifier cmdIn
  *                      
  * @return EP_SUCC_            Read Successful
  *         EP_FAIL_            cmd does not exist in the object dictionary
  *         EP_MUTEX_LOCKED_    Read permission denied.
  */
int EasyObjectDictionary::Read(
    EP_CMD_TYPE_ cmdIn,      ///< [INPUT]  Identifier to specify which object in the dictionary
    char**       dataOut,    ///< [OUTPUT] Pointer to the object
    int*         lengthOut   ///< [OUTPUT] Length (size) of the object
){
    // Mutex Protection:
    int key = EOD_DB_FindKey(cmdIn);
    if( key == -1) {
        return EP_FAIL_;
    }
    else{
        if((eOD_DB_Dynamic[key].mutex) & EOD_MUTEX_READ_PROTECT_) 
            return EP_MUTEX_LOCKED_;
        if(eOD_DB_Dynamic[key].mutex & EOD_MUTEX_LOCKED_) 
            return EP_MUTEX_LOCKED_;
    }
    
    // Enable ReadProtection:
    eOD_DB_Dynamic[key].mutex |= EOD_MUTEX_READ_PROTECT_;
       
    // Read data:
    *dataOut   = (char*)(eOD_DB_Dynamic[key].data);
    *lengthOut =  eOD_DB_Static[key].size;
    return EP_SUCC_;
}


/**
  *  Read Only the header
  * @return EP_SUCC_            Read Successful
  *         EP_FAIL_            cmd does not exist in the object dictionary
  *         EP_MUTEX_LOCKED_    Read permission denied.
 */
int EasyObjectDictionary::Read_Header(
    EP_CMD_TYPE_ cmdIn,    ///< [INPUT]   Specify the Object where to read the header
    Ep_Header* headerOut   ///< [OUTPUT]  Output the header
){
    // Mutex Protection:
    int key = EOD_DB_FindKey(cmdIn);
    if( key == -1) {
        return EP_FAIL_;
    }
    else{
        if((eOD_DB_Dynamic[key].mutex) & EOD_MUTEX_READ_PROTECT_) 
            return EP_MUTEX_LOCKED_;
        if(eOD_DB_Dynamic[key].mutex & EOD_MUTEX_LOCKED_) 
            return EP_MUTEX_LOCKED_;
    }
    
    // Enable ReadProtection:
    // EOD_DB_SetReadProtect(cmdIn, true); // Same effect as below, but has lower efficiency in execution
    eOD_DB_Dynamic[key].mutex |= EOD_MUTEX_READ_PROTECT_;
        
    // Read data:
    char* dataTmp   = (char*)(eOD_DB_Dynamic[key].data);
    for(int i=0; i<((signed)(sizeof(Ep_Header))); i++){
        *((char*)(headerOut) + i) = *(dataTmp + i);
    }
    
    // EOD_DB_SetReadProtect(cmdIn, false); // Same effect as below, but has lower efficiency in execution
    eOD_DB_Dynamic[key].mutex &= (~EOD_MUTEX_READ_PROTECT_); 
    return EP_SUCC_;       
}

/**
  * Analysis the input byte-array and fetch the "cmd" and "fromId" from the stream.
  * @return EP_SUCC_            Read successfully
  *         EP_FAIL_            lengthIn doesn't match the type size, or the cmd does not exist in the object dictionary
  */
int EasyObjectDictionary::Read_Header(
    char*         dataIn,    ///< [INPUT] The data array
    int           lengthIn,  ///< [INPUT] Size of the input data array
    Ep_Header*    headerOut  ///< [OUTPUT] interpreted header from the data array.
){
    if(lengthIn < (signed)(sizeof(Ep_Header))) return EP_FAIL_;
    for(int i=0; i< ((signed)sizeof(Ep_Header)); i++){
        *((char*)(headerOut)+i) = dataIn[i];
    }
    return EP_SUCC_;
}
//------------------------------------------------------------------------
//                                 Read
//------------------------------------------------------------------------









//------------------------------------------------------------------------
//                                 Write
//------------------------------------------------------------------------
/**
  * Copy data in a byte array into the relevant structure specified by the cmd at the beginning of the byte array.
  * @return EP_SUCC_            Write successfully
  *         EP_FAIL_            lengthIn doesn't match the type size, or the cmd does not exist in the object dictionary
  *         EP_MUTEX_LOCKED_    Write permission denied.
  */
int EasyObjectDictionary::Write(
    char*         dataIn,    ///< [INPUT] The data array
    int           lengthIn,  ///< [INPUT] Size of the input data array
    Ep_Header*    headerOut  ///< [OUTPUT] interpreted cmd from the data array.
){
    int retVal = EP_SUCC_;
    
    // Fetch header form the data array:
    if(EP_FAIL_ == Read_Header(dataIn, lengthIn, headerOut)){
        return EP_FAIL_;
    }

    // Mutex Protection:
    int key = EOD_DB_FindKey(headerOut->cmd);
    if( key == -1) {
        return EP_FAIL_;
    }
    else{
        if((eOD_DB_Dynamic[key].mutex) & EOD_MUTEX_LOCKED_) 
            return EP_MUTEX_LOCKED_;
        if((eOD_DB_Dynamic[key].mutex) & EOD_MUTEX_WRITE_PROTECT_) 
            return EP_MUTEX_LOCKED_;
        if((eOD_DB_Dynamic[key].mutex) & EOD_MUTEX_READ_PROTECT_) 
            return EP_MUTEX_LOCKED_;
    }
    eOD_DB_Dynamic[key].mutex |= EOD_MUTEX_LOCKED_;

    // Write Operation:
    if(lengthIn == eOD_DB_Static[key].size){
        char* dataPtr = (char*)(eOD_DB_Dynamic[key].data);
        for(int i=0; i<lengthIn; i++){
          *(dataPtr + i) = *(dataIn + i);
        }
    }
    else{
         retVal = EP_FAIL_;
    }

    // Mutex: clear EOD_MUTEX_LOCKED_ bit:
    eOD_DB_Dynamic[key].mutex &= (~EOD_MUTEX_LOCKED_); 
    return retVal;
}

/** Overwrite new header into Object specified by cmd
  * @return EP_SUCC_            Write successfully
  *         EP_FAIL_            lengthIn doesn't match the type size, or the cmd does not exist in the object dictionary
  *         EP_MUTEX_LOCKED_    Write permission denied.
  */
int EasyObjectDictionary::Write_Header(
    EP_CMD_TYPE_ cmdIn,    ///< [INPUT]  Specify which Object to overwrite the new header
    Ep_Header headerIn     ///< [INPUT]  The new header
){
    int retVal = EP_SUCC_;
        
    // Mutex Protection:
    int key = EOD_DB_FindKey(cmdIn);
    if( key == -1) {
        return EP_FAIL_;
    }
    else{
        if((eOD_DB_Dynamic[key].mutex) & EOD_MUTEX_LOCKED_) 
            return EP_MUTEX_LOCKED_;
        if((eOD_DB_Dynamic[key].mutex) & EOD_MUTEX_WRITE_PROTECT_) 
            return EP_MUTEX_LOCKED_;
        if((eOD_DB_Dynamic[key].mutex) & EOD_MUTEX_READ_PROTECT_) 
            return EP_MUTEX_LOCKED_;
    }
    eOD_DB_Dynamic[key].mutex |= EOD_MUTEX_LOCKED_;

    // Write Operation :
    char* dataPtr = (char*)(eOD_DB_Dynamic[key].data);
    for(int i=0; i<((signed)(sizeof(Ep_Header))); i++){
      *(dataPtr + i) = *((char*)(&headerIn) + i);
    }
    
    // Mutex: clear EOD_MUTEX_LOCKED_ bit:
    eOD_DB_Dynamic[key].mutex &= (~EOD_MUTEX_LOCKED_); 
    return retVal; 
}


/**
 * Set the toId of an Object specified by the cmd without changing other contents
 * @return EP_SUCC_            Write successfully
 *         EP_FAIL_            fail
 *         EP_MUTEX_LOCKED_    Permission denied.
 */
int EasyObjectDictionary::Write_Header_toId(
    EP_CMD_TYPE_    cmdIn,     ///< [INPUT]
    EP_ID_TYPE_    toIdIn      ///< [INPUT]
){
    Ep_Header header;
    int retVal;
    retVal = Read_Header(cmdIn, &header);
    if(retVal == EP_SUCC_){
        header.toId = toIdIn;
        retVal = Write_Header(cmdIn, header);
    }
    return retVal;
}

//------------------------------------------------------------------------
//                                 Write
//------------------------------------------------------------------------


