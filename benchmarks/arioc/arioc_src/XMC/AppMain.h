/*
  AppMain.h

    Copyright (c) 2015-2019 Johns Hopkins University.  All rights reserved.

    This file is part of the Arioc software distribution.  It is subject to the license terms
    in the LICENSE.txt file found in the top-level directory of the Arioc software distribution.
    The contents of this file, in whole or in part, may only be copied, modified, propagated, or
    redistributed in accordance with the license terms contained in LICENSE.txt.
*/
#pragma once
#define __AppMain__

#pragma region structs
struct SamFilePartitionInfo
{
    INT64   ofsStart;
    INT64   ofsLimit;

    INT64   TotalRows;
    INT64   TotalXM;
    INT64   Cm_CpG;
    INT64   Cm_CHG;
    INT64   Cm_CHH;
    INT64   Cm_unknown;
    INT64   C_CpG;
    INT64   C_CHG;
    INT64   C_CHH;
    INT64   C_unknown;

    INT64   MapqCount[256];

    SamFilePartitionInfo() : ofsStart(0), ofsLimit(0),
                             TotalRows(0), TotalXM(0),
                             Cm_CpG(0), Cm_CHG(0), Cm_CHH(0), Cm_unknown(0),
                             C_CpG(0), C_CHG(0), C_CHH(0), C_unknown(0)
    {
        memset( this->MapqCount, 0, sizeof this->MapqCount );
    }

    void AccumulateCounts( SamFilePartitionInfo& _other )
    {
        this->TotalRows += _other.TotalRows;
        this->TotalXM += _other.TotalXM;
        this->Cm_CpG += _other.Cm_CpG;
        this->Cm_CHG += _other.Cm_CHG;
        this->Cm_CHH += _other.Cm_CHH;
        this->Cm_unknown += _other.Cm_unknown;
        this->C_CpG += _other.C_CpG;
        this->C_CHG += _other.C_CHG;
        this->C_CHH += _other.C_CHH;
        this->C_unknown += _other.C_unknown;

        for( size_t n=0; n<arraysize(this->MapqCount); ++n )
            this->MapqCount[n] += _other.MapqCount[n];
    }
};
#pragma endregion

/// <summary>
/// Class <c>AppMain</c> parses input parameters, launches the application, and displays performance metrics.
/// </summary>
class AppMain : public AppMainCommon
{
    private:
        HiResTimer  m_hrt;

    public:
        INT32                   ConcurrentThreadCount;
        char                    SamFilename[FILENAME_MAX];
        SamFilePartitionInfo    SFPI;

        INT32   Filter_f;       // user-specified parameters (see AppGlobal.cpp)
        INT32   Filter_F;
        INT32   Filter_G;
        INT32   Filter_q;
        bool    Report_d;
        bool    Report_h;

    protected:
        virtual void parseXmlElements( void );

    public:
        AppMain( void );
        virtual ~AppMain( void );
        void Launch( void );
        virtual void LoadConfig( void );
};
