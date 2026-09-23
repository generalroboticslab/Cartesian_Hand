#include <iostream>

#ifdef _WIN32
#include <windows.h>
#define imsleep(microsecond) Sleep(microsecond) // ms
#else
#include <unistd.h>
#define imsleep(microsecond) usleep(1000 * microsecond) // ms
#endif

#include <vector>
#include <mutex>

#include "CSerialPort/SerialPort.h"
#include "CSerialPort/SerialPortInfo.h"
using namespace itas109;

// Step 1:  Include Library Headers:
#include "EasyProfile/EasyObjectDictionary.h"
#include "EasyProfile/EasyProfile.h"
// Note: eOD and eP are now instance members of the IMU class (not global)

#include <Eigen/Dense>
#include <Eigen/Geometry>
#include <memory>
template <typename ScalarType, int NumElements>
using VectorNd = Eigen::Matrix<ScalarType, NumElements, 1>;

std::string char2hexstr(const char *str, int len)
{
    static const char hexTable[17] = "0123456789ABCDEF";

    std::string result;
    for (int i = 0; i < len; ++i)
    {
        result += "0x";
        result += hexTable[(unsigned char)str[i] / 16];
        result += hexTable[(unsigned char)str[i] % 16];
        result += " ";
    }
    return result;
}



/**
 * @brief Converts a quaternion in xyzw format to a 3x3 rotation matrix using Eigen.
 */
Eigen::Matrix3f quat_to_rotation_matrix(const VectorNd<float, 4>& quat_xyzw) {
    // Eigen::Quaternionf ctor order: (w, x, y, z)
    return Eigen::Quaternionf(quat_xyzw(3), quat_xyzw(0), quat_xyzw(1), quat_xyzw(2)).toRotationMatrix();
}



int countRead = 0;

class IMU : public CSerialPortListener
{
public:
    CSerialPort sp;
    std::string portName = "/dev/ttyACM0";

    // Per-instance EasyProfile objects (not global, so multiple IMUs work correctly)
    EasyObjectDictionary eOD;
    EasyProfile eP{&eOD};

    bool should_print = false;

    int counter = 0;

    char rxBuffer[1024];
    int rxSize;
    VectorNd<float, 3> raw_acc;     // raw aceleration [g] (contains gravity)
    VectorNd<float, 3> ang_vel;     // raw angular velocity, in local frame!!!
    VectorNd<float, 3> world_space_ang_vel;   // raw angular velocity in world frame
    VectorNd<float, 3> raw_mag;     // raw magnetic field
    VectorNd<float, 4> quat_xyzw;   // quaternion in xyzw format [x, y, z, w]
    VectorNd<float, 4> quat_wxyz;   // quaternion in wxyz format [w, x, y, z]
    VectorNd<float, 3> euler;       // roll, pitch, yaw (earth frame)
    VectorNd<float, 3> gravity_vec; // gravity vector [g]
    VectorNd<float, 3> direct_gravity_vec; // gravity vector dirctly from IMU
    Eigen::Matrix3f rotation_matrix; // rotation matrix computed from quaternion

    uint32_t timeStamp;
    uint16_t qos;
    int8_t temperature;
    uint16_t updateRate;

    IMU() {}
    ~IMU() {
        close();
    }
    IMU(std::string portName)
        : portName(portName) {
          };

    void close() {
        printf("\033[1;32m[%s:%d] IMU: Close %s\033[0m\n", __FILE__, __LINE__, portName.c_str());
        sp.setReadIntervalTimeout(1);
        sp.disconnectReadEvent();
        if (sp.isOpen()){
            sp.close();
        }
    }

    bool findPortNameByDescription(std::string portDescription = "STMicroelectronics Virtual COM Port")
    {
        std::vector<SerialPortInfo> ports = CSerialPortInfo::availablePortInfos();

        if (should_print){
            printf("AvailableFriendlyPorts:\n");
            for (int i = 0; i < ports.size(); ++i)
                printf("    %d - %s %s %s\n", i, ports[i].portName, ports[i].description, ports[i].hardwareId);
        }
        
        for (int i = 0; i < ports.size(); ++i)
        {
            std::string _port_description = ports[i].description;
            if ( _port_description == portDescription){
                this->portName = ports[i].portName;
                if (should_print)
                    printf("\033[1;32m[matched] Port Name: %s\033[0m\n", ports[i].portName); 
                return true;
            }
        }
        if (should_print)
            printf("\033[31mIMU: No matching port description found.\033[0m\n");
        return false;
    }


    void run()
    {
        raw_acc.setZero();
        ang_vel.setZero();
        raw_mag.setZero();
        quat_xyzw.setZero();
        euler.setZero();
        gravity_vec.setZero();
        direct_gravity_vec.setZero();


        // use lsusb to find COM port description

        sp.init(portName.c_str(),    // windows:COM1 Linux:/dev/ttyS0
                                     // itas109::BaudRate115200, // baudrate
                4000000,             // baudrate
                itas109::ParityNone, // parity
                itas109::DataBits8,  // data bit
                itas109::StopOne,    // stop bit
                itas109::FlowNone,   // flow
                256                 // read buffer size
        );
        int ret;
        sp.setReadIntervalTimeout(0); // read interval timeout 0ms
        sp.setMinByteReadNotify(8); // set minimum byte of read notify
        ret = sp.open();
        // sp.flushBuffers();
        if (should_print){
            printf("Version: %s\n\n", sp.getVersion());
            printf("\033[1;32mIMU: Open %s %s\033[0m\n", portName.c_str(), sp.isOpen() ? "Success" : "Failed");
            printf("\033[1;32mIMU: Code: %d, Message: %s\033[0m\n", sp.getLastError(), sp.getLastErrorMsg());

        }

        ret = sp.connectReadEvent(this);
        if (ret != 0)
        {
            printf("\033[31mIMU: Failed to connect read event!\033[0m\n");
            return;
        }

        int readIntervalTimeout = sp.getReadIntervalTimeout();
        printf("\033[1;32mIMU: Read Interval Timeout: %d\033[0m\n", readIntervalTimeout);
    }

    // Parse the main combo packet from the IMU — updates all sensor fields under lock
    void parse_combo(const Ep_Combo& ep_Combo)
    {
        std::lock_guard<std::mutex> guard(data_mutex_);
        timeStamp = ep_Combo.timeStamp;             // timeStamp    unit: uS
        // uint32_t deviceId = ep_Combo.header.fromId; // The Node ID of the TransducerM that transmitted this ep_Combo
        qos = ep_Combo.sysState.bits.qos;           // Quality-of-Service  possible values: 0,1,2,3,4,5
        temperature = ep_Combo.temperature;         // temperature  unit: Celcius
        updateRate = (ep_Combo.updateRate) * 10;    // updateRate   unit: Hz
        // Accelerometer:
        // need to negate to achive when imu is perfectly horizonal and z axis is pointing upward, expect [0,0,g]
        raw_acc(0) = -(ep_Combo.ax) * (1e-5f); // Unit: 1g, 1g = 9.794m/(s^2)
        raw_acc(1) = -(ep_Combo.ay) * (1e-5f);
        raw_acc(2) = -(ep_Combo.az) * (1e-5f);
        // Gyroscope:
        ang_vel(0) = (ep_Combo.wx) * (1e-5f); // Unit: rad/s
        ang_vel(1) = (ep_Combo.wy) * (1e-5f);
        ang_vel(2) = (ep_Combo.wz) * (1e-5f);
        // Magnetometer:
        raw_mag(0) = (ep_Combo.mx) * (1e-3f); // Unit: one earth magnetic field
        raw_mag(1) = (ep_Combo.my) * (1e-3f); // vector (mx, my, mz) is used as direction reference of the local magnetic field.
        raw_mag(2) = (ep_Combo.mz) * (1e-3f); // The norm(mx, my, mz) may not be accurate.
        // IMU sends wxyz: q1=w, q2=x, q3=y, q4=z
        const float qw = (ep_Combo.q1) * (1e-7f);
        const float qx = (ep_Combo.q2) * (1e-7f);
        const float qy = (ep_Combo.q3) * (1e-7f);
        const float qz = (ep_Combo.q4) * (1e-7f);
        // Skip orientation update on corrupt packets; keeps last valid quat/rotation/gravity.
        // Check sum-of-squares ≈ 1 (avoids sqrt; threshold 0.2 ≈ |norm-1| < 0.1)
        if (std::abs(qw*qw + qx*qx + qy*qy + qz*qz - 1.0f) < 0.2f) {
            quat_wxyz << qw, qx, qy, qz;
            quat_xyzw << qx, qy, qz, qw;
            rotation_matrix = quat_to_rotation_matrix(quat_xyzw);
            world_space_ang_vel = rotation_matrix * ang_vel;
            gravity_vec = -rotation_matrix.row(2);
        }

        // RPY:
        euler(0) = ((float)(ep_Combo.roll)) * (1e-2f);  // Unit: degree
        euler(1) = -((float)(ep_Combo.pitch)) * (1e-2f); // Unit: degree
        euler(2) = -((float)(ep_Combo.yaw)) * (1e-2f);   // Unit: degree

        // float ax = (ep_Combo.ax) * (1e-5f); // Unit: 1g, 1g = 9.794m/(s^2)
        // float ay = (ep_Combo.ay) * (1e-5f);
        // float az = (ep_Combo.az) * (1e-5f);
        // // Gyroscope:
        // float wx = (ep_Combo.wx) * (1e-5f); // Unit: rad/s
        // float wy = (ep_Combo.wy) * (1e-5f);
        // float wz = (ep_Combo.wz) * (1e-5f);
        // // Magnetometer:
        // float mx = (ep_Combo.mx) * (1e-3f); // Unit: one earth magnetic field
        // float my = (ep_Combo.my) * (1e-3f); // vector (mx, my, mz) is used as direction reference of the local magnetic field.
        // float mz = (ep_Combo.mz) * (1e-3f); // The norm(mx, my, mz) may not be accurate.
        // // Quaternion in (w,x,y,z) format
        // float q1 = (ep_Combo.q1) * (1e-7f);
        // float q2 = (ep_Combo.q2) * (1e-7f);
        // float q3 = (ep_Combo.q3) * (1e-7f);
        // float q4 = (ep_Combo.q4) * (1e-7f);
        // // RPY:
        // float roll = (ep_Combo.roll) * (1e-2f);   // Unit: degree
        // float pitch = (ep_Combo.pitch) * (1e-2f); // Unit: degree
        // float yaw = (ep_Combo.yaw) * (1e-2f);     // Unit: degree

        if (should_print)
        {
            printf("qos       %d\n", qos);
            printf("temp      %d\n", temperature);
            printf("time      %d\n", timeStamp);
            printf("update    %d\n", updateRate);
            printf("RawAcc    %+9.4f %+9.4f %+9.4f \n", raw_acc(0), raw_acc(1), raw_acc(2));
            printf("RawGyro   %+9.4f %+9.4f %+9.4f \n", ang_vel(0), ang_vel(1), ang_vel(2));
            printf("RawMag    %+9.4f %+9.4f %+9.4f \n", raw_mag(0), raw_mag(1), raw_mag(2));
            printf("Q(xyzw)   %+9.4f %+9.4f %+9.4f %+9.4f\n", quat_xyzw(0), quat_xyzw(1), quat_xyzw(2), quat_xyzw(3));
            printf("RPY       %+9.4f %+9.4f %+9.4f\n", euler(0), euler(1), euler(2));
            printf("Gravity   %+9.4f %+9.4f %+9.4f \n", gravity_vec(0), gravity_vec(1), gravity_vec(2));
            printf("AngVel_b  %+9.4f %+9.4f %+9.4f \n", ang_vel(0), ang_vel(1), ang_vel(2));
            printf("AngVel_w  %+9.4f %+9.4f %+9.4f \n", world_space_ang_vel(0), world_space_ang_vel(1), world_space_ang_vel(2));
        }

        counter++;

        // // Simple checksum:
        // uint16_t checksum = 0;
        // for (unsigned int i = 0; i < (sizeof(Ep_Combo) - 2); i++)
        // {
        //     checksum += (*(((uint8_t *)(&ep_Combo)) + i));
        // }
        // if (checksum == (ep_Combo.simpleChecksum))
        // { // This is redundant check and is not necessary when SYD Dynamics communication library is used.
        //   // Checksum is correct                            // The simple checksum is designed to be used when user implements own communication library and do not wish to use the CRC checking field of the data stream.
        //     // Example use of data here...
        //     // printf("RawAcc  %+9.4f %+9.4f %+9.4f \n", ax, ay, az);
        //     // printf("RawGyro %+9.4f %+9.4f %+9.4f \n", wx, wy, wz);
        //     // printf("RawMag  %+9.4f %+9.4f %+9.4f \n", mx, my, mz);
        //     // printf("Q       %+9.4f %+9.4f %+9.4f %f\n", q1, q2, q3, q4);
        //     // printf("RPY     %+9.4f %+9.4f %+9.4f\n", roll, pitch, yaw);
        // }
        // else
        // {
        //     printf("Checksum Error\n");
        // }
    }

    void onReadEvent(const char *portName, unsigned int readBufferLen)
    {
        if (readBufferLen <= 0) return;

        rxSize = sp.readData(rxBuffer, readBufferLen);
        if (rxSize > 0)
        {
            // printf("%s - Count: %d, Length: %d, Str: %s, Hex: %s\n", portName, ++countRead, rxSize, rxBuffer, char2hexstr(rxBuffer, rxSize).c_str());
            OnSerialRX();
        }
    };

    void OnSerialRX(void)
    {
        // char rxBuffer[64];
        char *rxData;
        // int rxSize;
        // SerialPort_ReadData(&rxBuffer, &rxSize); // Read from serial port buffer. Please modify according to your platform.
        rxData = rxBuffer;

        Ep_Header header; // Then let the EasyProfile do the rest such as data assembling and checksum verification.
        if (EP_SUCC_ == eP.On_RecvPkg(rxData, rxSize, &header))
        {
            uint32 fromId = header.fromId;
            (void)fromId;
            switch (header.cmd)
            { // The program will only reach this line if and only if a correct and complete package has received.
            case EP_CMD_ACK_:
            {
                Ep_Ack ep_Ack; //           tasks for different types of data.
                if (EP_SUCC_ == eOD.Read_Ep_Ack(&ep_Ack))
                {
                }
            }
            break;
            case EP_CMD_STATUS_:
            {
                Ep_Status ep_Status;
                if (EP_SUCC_ == eOD.Read_Ep_Status(&ep_Status))
                {
                }
            }
            break;
            case EP_CMD_COMBO_:
            {
                Ep_Combo ep_Combo;
                if (EP_SUCC_ == eOD.Read_Ep_Combo(&ep_Combo))
                    parse_combo(ep_Combo);
            }
            break;
            case EP_CMD_Raw_GYRO_ACC_MAG_:
            { // Here we demonstrate a few examples on how to use the received data
                Ep_Raw_GyroAccMag ep_Raw_GyroAccMag;
                if (EP_SUCC_ == eOD.Read_Ep_Raw_GyroAccMag(&ep_Raw_GyroAccMag))
                {
                    // Raw Data received
                    unsigned int timeStamp = ep_Raw_GyroAccMag.timeStamp;
                    float ax = ep_Raw_GyroAccMag.acc[0]; // Note 1: ep_Raw_GyroAccMag is defined in the EasyProfile library as a global variable
                    float ay = ep_Raw_GyroAccMag.acc[1]; // Note 2: for the units and meaning of each value, refer to EasyObjectDictionary.h
                    float az = ep_Raw_GyroAccMag.acc[2];
                    float wx = ep_Raw_GyroAccMag.gyro[0];
                    float wy = ep_Raw_GyroAccMag.gyro[1];
                    float wz = ep_Raw_GyroAccMag.gyro[2];
                    float mx = ep_Raw_GyroAccMag.mag[0];
                    float my = ep_Raw_GyroAccMag.mag[1];
                    float mz = ep_Raw_GyroAccMag.mag[2];

                    if (should_print)
                    {
                        printf("[should not see this] EP_CMD_Raw_GYRO_ACC_MAG_:\n");
                        printf("[should not see this] RawAcc    %+9.4f %+9.4f %+9.4f \n", ax, ay, az);
                        printf("[should not see this] RawGyro   %+9.4f %+9.4f %+9.4f \n", wx, wy, wz);
                        printf("[should not see this] RawMag    %+9.4f %+9.4f %+9.4f \n", mx, my, mz);
                    }
                }
            }
            break;
            case EP_CMD_Q_S1_E_:
            {
                Ep_Q_s1_e ep_Q_s1_e;
                if (EP_SUCC_ == eOD.Read_Ep_Q_s1_e(&ep_Q_s1_e))
                {
                    // Quanternion received
                    unsigned int timeStamp = ep_Q_s1_e.timeStamp;
                    float q0 = ep_Q_s1_e.q[0]; // Note 1, ep_Q_s1_e is defined in the EasyProfile library as a global variable
                    float q1 = ep_Q_s1_e.q[1]; // Note 2, for the units and meaning of each value, refer to EasyObjectDictionary.h
                    float q2 = ep_Q_s1_e.q[2];
                    float q3 = ep_Q_s1_e.q[3];
                    // used q1 q2 q3 q4 here...
                    // ...
                    if (should_print)
                    {
                        printf("[should not see this] Q [earth] %+9.4f %+9.4f %+9.4f %f\n", q0, q1, q2, q3);
                    }
                }
            }
            break;
            case EP_CMD_EULER_S1_E_:

            {
                Ep_Euler_s1_e ep_Euler_s1_e;
                if (EP_SUCC_ == eOD.Read_Ep_Euler_s1_e(&ep_Euler_s1_e))
                {
                }
            }
            break;
            case EP_CMD_RPY_:
            {
                Ep_RPY ep_RPY;
                if (EP_SUCC_ == eOD.Read_Ep_RPY(&ep_RPY))
                {
                    // Roll Pitch Yaw data received
                    unsigned int timeStamp = ep_RPY.timeStamp;
                    float roll = ep_RPY.roll;   // Note 1, ep_RPY is defined in the EasyProfile library as a global variable
                    float pitch = ep_RPY.pitch; // Note 2, for the units and meaning of each value, refer to EasyObjectDictionary.h
                    float yaw = ep_RPY.yaw;
                    // Use roll, pitch, yaw data here...
                    // ...
                    if (should_print)
                    {
                        printf("[should not see this] RPY %+9.4f %+9.4f %+9.4f\n", roll, pitch, yaw);
                    }
                }
            }
            break;
            case EP_CMD_GRAVITY_:
            {
                Ep_Gravity ep_Gravity;
                if (EP_SUCC_ == eOD.Read_Ep_Gravity(&ep_Gravity))
                {
                    std::lock_guard<std::mutex> guard(data_mutex_);
                    uint32 timeStamp = ep_Gravity.timeStamp;             // timeStamp    unit: uS
                    direct_gravity_vec(0) = ep_Gravity.g[0];
                    direct_gravity_vec(1) = ep_Gravity.g[1];
                    direct_gravity_vec(2) = ep_Gravity.g[2];

                    if (should_print)
                    {
                        // printf("Gravity:%+9.4f %+9.4f %+9.4f\n", ep_Gravity.g[0], ep_Gravity.g[1], ep_Gravity.g[2]);
                        printf("[should not see this] Gravity %+9.4f %+9.4f %+9.4f\n", direct_gravity_vec(0), direct_gravity_vec(1), direct_gravity_vec(2));
                    }
                }
            }
            break;
            }
        
        
        }

        else
        {
            printf("Count: %d, Length: %d, Str: %s, Hex: %s\n", ++countRead, rxSize, rxBuffer, char2hexstr(rxBuffer, rxSize).c_str());
        }
    }

private:
    // Helper: apply rotation to vector
    template<typename Vec>
    Vec apply_rotation(const Vec& v) const { return rotation_offset_ ? (*rotation_offset_) * v : v; }

    // Helper: convert rotation matrix to euler angles in degrees
    static VectorNd<float, 3> matrix_to_euler_deg(const Eigen::Matrix3f& R) {
        float roll = std::atan2(R(2,1), R(2,2)), pitch = std::asin(std::clamp(-R(2,0), -1.0f, 1.0f)), yaw = std::atan2(R(1,0), R(0,0));
        if (yaw > 0) yaw -= 2.0f * M_PI;
        return VectorNd<float, 3>{roll, pitch, yaw} * (180.0f / M_PI);
    }

public:
    // Rotation offset management
    void set_rotation_offset(const Eigen::Matrix3f& offset) {
        rotation_offset_ = std::make_unique<Eigen::Matrix3f>(offset);
        cache_counter_ = 0;
    }
    void clear_rotation_offset() { rotation_offset_.reset(); cache_counter_ = 0; }
    Eigen::Matrix3f get_rotation_offset() const {
        return rotation_offset_ ? *rotation_offset_ : Eigen::Matrix3f::Identity();
    }
    bool has_rotation_offset() const { return rotation_offset_ != nullptr; }

    // Transformed rotation matrix (cached - reused by euler)
    Eigen::Matrix3f get_transformed_rotation_matrix() const {
        if (!rotation_offset_) return rotation_matrix;
        if (cache_counter_ != counter) {
            cached_transformed_matrix_ = (*rotation_offset_) * rotation_matrix * rotation_offset_->transpose();
            cache_counter_ = counter;
        }
        return cached_transformed_matrix_;
    }

    // Transformed quaternion (similarity transform)
    VectorNd<float, 4> get_transformed_quat_xyzw() const {
        if (!rotation_offset_) return quat_xyzw;
        Eigen::Quaternionf q = (Eigen::Quaternionf(*rotation_offset_) *
            Eigen::Quaternionf(quat_xyzw(3), quat_xyzw(0), quat_xyzw(1), quat_xyzw(2)) *
            Eigen::Quaternionf(*rotation_offset_).conjugate()).normalized();
        return VectorNd<float, 4>{q.x(), q.y(), q.z(), q.w()};
    }

    // Transformed quaternion in wxyz format
    VectorNd<float, 4> get_transformed_quat_wxyz() const {
        if (!rotation_offset_) return quat_wxyz;
        Eigen::Quaternionf q = (Eigen::Quaternionf(*rotation_offset_) *
            Eigen::Quaternionf(quat_xyzw(3), quat_xyzw(0), quat_xyzw(1), quat_xyzw(2)) *
            Eigen::Quaternionf(*rotation_offset_).conjugate()).normalized();
        return VectorNd<float, 4>{q.w(), q.x(), q.y(), q.z()};
    }

    // Transformed euler (reuses cached matrix)
    VectorNd<float, 3> get_transformed_euler() const {
        return rotation_offset_ ? matrix_to_euler_deg(get_transformed_rotation_matrix()) : euler;
    }

    // Transformed vectors (simple rotation)
    // "transformed" = mounting-offset correction only; ang_vel stays in BODY frame (raw gyro).
    // Use get_transformed_world_space_ang_vel() if you need world frame.
    VectorNd<float, 3> get_transformed_ang_vel() const { return apply_rotation(ang_vel); }
    VectorNd<float, 3> get_transformed_world_space_ang_vel() const { return apply_rotation(world_space_ang_vel); }
    VectorNd<float, 3> get_transformed_raw_acc() const { return apply_rotation(raw_acc); }
    VectorNd<float, 3> get_transformed_gravity_vec() const { return apply_rotation(gravity_vec); }
    VectorNd<float, 3> get_transformed_raw_gravity_vec() const { return apply_rotation(direct_gravity_vec); }
    VectorNd<float, 3> get_transformed_raw_mag() const { return apply_rotation(raw_mag); }

    // Protects sensor fields from torn reads during callback writes
    mutable std::mutex data_mutex_;

private:
    // Rotation offset support
    std::unique_ptr<Eigen::Matrix3f> rotation_offset_;
    mutable Eigen::Matrix3f cached_transformed_matrix_;
    mutable uint32_t cache_counter_ = 0;
};
