/**
 * EasyProfile.cpp
 * @author COPYRIGHT(c) 2017 SYD Dynamics ApS
 * @see    EasyProfile.h for more descriptions.
 */
#include "EasyProfile.h"


EasyProfile::EasyProfile(EasyObjectDictionary *eOD_i){
    // Easy Object Dictionary:
    eOD = eOD_i;
    
    // Easy Protocol:
    eP = new EasyProtocol();
    int maxSize;
    maxSize = eOD->Get_MaxSize();
    eP->Init( maxSize, maxSize );
}


EasyProfile::~EasyProfile(){
    if(eP) {delete eP; eP = 0;}
}


/**
 * @brief   EasyProfile::On_SendPkg
 * @return  EP_FAIL_         payloadSize too large or easyprotocol setting error.
 *          EP_MUTEX_LOCKED_ Read permission denied.
 *          EP_SUCC_         Creation of the new package done successfully.
 */
int EasyProfile::On_SendPkg(
        EP_CMD_TYPE_    txPkgCmd,  ///< [INPUT]   Specify the relevant data structre to be sent. (e.g. EP_CMD_REQUEST_)
        EP_ID_TYPE_*    toId,      ///< [OUTPUT]  Destination ID of the package to be sent.
        char**          pkgData,   ///< [OUTPUT]  The pointer to the generated package buffer.
        int*            pkgSize    ///< [OUTPUT]  The size of the generated package buffer.
){
    int   retVal;
    char* objData;
    int   objSize;
    retVal = eOD->Read(txPkgCmd, &objData, &objSize);
    if(retVal == EP_SUCC_){
        // Fetch toId:
        Ep_Header headerOut;
        retVal = eOD->Read_Header(objData, objSize, &headerOut);
        *toId = headerOut.toId;
        
        if(retVal == EP_SUCC_){
            retVal = eP->CreateOutputPackage( objData, objSize, pkgData, pkgSize);
        }
    }
    eOD->EOD_DB_SetWriteProtect(txPkgCmd, false);
    eOD->EOD_DB_SetReadProtect(txPkgCmd, false);
    return retVal;
}



/**
 * @brief  EasyProfile::On_RecvPkg
 * @return EP_SUCC_          New package received
 *         EP_NORMAL_EXIT_   No new package received, no error occured / package ignored due to RecvFilter policy
 *         EP_MUTEX_LOCKED_  Read / Write permission denied
 *         EP_FAIL_          Bad package / Bad hadder received
 */
int EasyProfile::On_RecvPkg(
        char*             data,    ///< [INPUT]  Pointer to received raw data
        int            dataSize,   ///< [INPUT]  Received raw data size
        Ep_Header       *header    ///< [OUTPUT] return the header of when new package received.
){
    char* payloadData;
    int   payloadSize;
    int        retVal;
    retVal = eP->AssembleInputPackage( data, dataSize, &payloadData, &payloadSize);

    if(retVal == EP_SUCC_){
        retVal = eOD->Write(payloadData, payloadSize, header);
        //if(retVal == EP_SUCC_){
        //    On_EasyProfile_RX(this, *header);
        //}
    }
    return retVal;
}


