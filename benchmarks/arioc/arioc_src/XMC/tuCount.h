/*
  tuCount.h

    Copyright (c) 2015-2019 Johns Hopkins University.  All rights reserved.

    This file is part of the Arioc software distribution.  It is subject to the license terms
    in the LICENSE.txt file found in the top-level directory of the Arioc software distribution.
    The contents of this file, in whole or in part, may only be copied, modified, propagated, or
    redistributed in accordance with the license terms contained in LICENSE.txt.
*/
#pragma once
#define __tuCount__

// forward definition
class XMC;

/// <summary>
/// Class <c>tuCount</c> counts cytosine instances.
/// </summary>
class tuCount : public tuBaseA
{
    private:
        const INT64             SAMFILEBUFSIZE = 10*1024*1024;
        const INT32             MINSAMRECORDSIZE = 100;

        XMC*                    m_pxmc;         // reference to the main XMC instance
        INT32                   m_ip;           // index of the SAM file partition handled in this tuCount instance
        SamFilePartitionInfo*   m_psfpi;        // pointer to this partition's SamFilePartitionInfo instance

        INT32                   m_filter_f;     // FLAG filter (include if all bits set)
        INT32                   m_filter_F;     // FLAG filter (include if no bits set)
        INT32                   m_filter_G;     // FLAG filter (exclude if all bits set)
        INT32                   m_filter_q;     // MAPQ filter (include if >=)
        bool                    m_report_d;     // report MAPQ distribution

        RaiiFile                m_fileSAM;      // SAM file
        WinGlobalPtr<char>      m_buf;          // SAM file input buffer
        INT32                   m_ofsBuf;       // offset of next byte of data in the buffer
        INT32                   m_cbBuf;        // number of bytes of data remaining in the buffer
        char*                   m_pBufLimit;    // pointer to the byte after the remaining data in the buffer

    protected:
        void main( void );

    private:
        tuCount( void );
        INT64 refreshInputBuffer( INT64 _ofs );
        INT64 parseSamRecord( INT64 _ofs );
        void parseXM( char* _p );

    public:
        tuCount( AppMain* _pam, XMC* _pxmc, INT32 _ip );
        virtual ~tuCount( void );
};

