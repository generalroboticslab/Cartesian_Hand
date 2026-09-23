/**
 * Easy Profile
 *
 * @brief  Managing the user-logic for sending and receiving Objects (items of the
 *         Object Dictionary (OD)). 
 *
 * @version 1.1.9     - Jul 02, 2016
 *          1.1.9 (R) - Jul 27, 2016 - Release Version
 * @attention
 * <h2><center>&copy; COPYRIGHT(c) 2017 SYD Dynamics ApS</center></h2>
 *
 * Redistribution and use in source and binary forms, with or without modification,
 * are permitted provided that the following conditions are met:
 *   1. Redistributions of source code must retain the above copyright notice,
 *      this list of conditions and the following disclaimer.
 *   2. Redistributions in binary form must reproduce the above copyright notice,
 *      this list of conditions and the following disclaimer in the documentation
 *      and/or other materials provided with the distribution.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
 * AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
 * DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
 * FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
 * DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
 * SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
 * CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
 * OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 * OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 */
#ifndef EASYPROFILE_H
#define EASYPROFILE_H

#include "BasicTypes.h"
#include "EasyProtocol.h"
#include "EasyObjectDictionary.h"
#ifdef EP_PLATFORM_QT5_
#include <QtCore>
#endif


#ifdef EP_PLATFORM_QT5_
class EasyProfile : public QObject {
    Q_OBJECT
#else
class EasyProfile {
#endif
public:
    EasyProfile(EasyObjectDictionary *eOD_i);
    ~EasyProfile();

// Send & Recv Pkg:
public:
    int   On_SendPkg(EP_CMD_TYPE_ txPkgCmd, EP_ID_TYPE_ *toId, char **pkgData, int *pkgSize);
    int   On_RecvPkg(char* data, int dataSize, Ep_Header* header);

// External Object Dictionary:
protected:
    EasyObjectDictionary* eOD;

// Easy Protocol:
protected:
    EasyProtocol *eP;

#ifdef EP_PLATFORM_QT5_
signals:
    void Sig_RX_Data(const Ep_Request& ep_Request);
    void Sig_RX_Data(const Ep_Ack& ep_Ack);
    void Sig_RX_Data(const Ep_Status& ep_Status);
    void Sig_RX_Data(const Ep_Raw_GyroAccMag& ep_Raw_GyroAccMag);
    void Sig_RX_Data(const Ep_Q_s1_s& ep_Q_s1_s);
    void Sig_RX_Data(const Ep_Q_s1_e& ep_Q_s1_e);
    void Sig_RX_Data(const Ep_Euler_s1_s& ep_Euler_s1_s);
    void Sig_RX_Data(const Ep_Euler_s1_e& ep_Euler_s1_e);
    void Sig_RX_Data(const Ep_RPY& ep_RPY);
    void Sig_RX_Data(const Ep_Gravity& ep_Gravity);
#endif
};


#endif // EASYPROFILE_H
