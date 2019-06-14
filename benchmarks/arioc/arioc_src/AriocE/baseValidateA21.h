/*
  baseValidateA21.h

    Copyright (c) 2015-2019 Johns Hopkins University.  All rights reserved.

    This file is part of the Arioc software distribution.  It is subject to the license terms
    in the LICENSE.txt file found in the top-level directory of the Arioc software distribution.
    The contents of this file, in whole or in part, may only be copied, modified, propagated, or
    redistributed in accordance with the license terms contained in LICENSE.txt.
*/
#pragma once
#define __baseValidateA21__

/// <summary>
/// Class <c>baseValidateA21</c> is the base implementation for validating encoded Q sequence data.
/// </summary>
class baseValidateA21 : public tuBaseA
{
    private:

    protected:
#pragma pack(push, 1)
        struct A21FIXEDFIELDS
        {
            UINT64  sqId;
            INT16   N;
            INT16   cb;
        };
#pragma pack(pop)

        static const INT32  A21BUFFERSIZE = (512 * 1024*1024);        // 512Mb
        static const INT32  m_cbFixed = sizeof( A21FIXEDFIELDS );
        AriocEncoderParams* m_psip;
        INT16               m_iInputFile;
        INT32               m_srcId;
        INT32               m_subId;
        RaiiSemaphore*      m_psemComplete;
        WinGlobalPtr<UINT8> m_inBuf;
        RaiiFile            m_inFile;

    protected:
        baseValidateA21( void );
        UINT8* getNext( INT32 cb );

    public:
        baseValidateA21( AriocEncoderParams* psip, SqCategory sqCat, INT16 iInputFile, RaiiSemaphore* psem );
        virtual ~baseValidateA21( void );
};
