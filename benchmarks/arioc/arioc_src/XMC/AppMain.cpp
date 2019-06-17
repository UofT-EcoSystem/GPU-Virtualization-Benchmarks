/*
  AppMain.cpp

    Copyright (c) 2015-2019 Johns Hopkins University.  All rights reserved.

    This file is part of the Arioc software distribution.  It is subject to the license terms
    in the LICENSE.txt file found in the top-level directory of the Arioc software distribution.
    The contents of this file, in whole or in part, may only be copied, modified, propagated, or
    redistributed in accordance with the license terms contained in LICENSE.txt.
*/
#include "stdafx.h"

#pragma region constructor/destructor
/// <summary>
/// default constructor
/// </summary>
AppMain::AppMain() : ConcurrentThreadCount(0),
                     Filter_f(0), Filter_F(0), Filter_G(0), Filter_q(0), Report_d(false), Report_h(false)
{
    memset( this->SamFilename, 0, sizeof this->SamFilename );
}

/// destructor
AppMain::~AppMain()
{
}
#pragma endregion

#pragma region virtual method implementations
/// [protected] method parseXmlElements
void AppMain::parseXmlElements()
{
    throw new ApplicationException( __FILE__, __LINE__, "%s: not implemented", __FUNCTION__ );
}

/// [public] method LoadConfig
void AppMain::LoadConfig()
{
    throw new ApplicationException( __FILE__, __LINE__, "%s: not implemented", __FUNCTION__ );
}

/// [protected] method Launch
void AppMain::Launch()
{
    // set up an XMC instance for the specified parameters
    XMC xmc( this );

    // echo the configuration parameters
    CDPrint( cdpCD0, " SAM file             : %s", this->SamFilename );
    CDPrint( cdpCD0, " Parameters:" );
    CDPrint( cdpCD0, "  -f                  : %d (0x%04x)", this->Filter_f, this->Filter_f );
    CDPrint( cdpCD0, "  -F                  : %d (0x%04x)", this->Filter_F, this->Filter_F );
    CDPrint( cdpCD0, "  -G                  : %d (0x%04x)", this->Filter_G, this->Filter_G );
    CDPrint( cdpCD0, "  -q                  : %u", this->Filter_q );
    CDPrint( cdpCD0, "  -p                  : %d thread%s", this->ConcurrentThreadCount, PluralS(this->ConcurrentThreadCount) );
    CDPrint( cdpCD0, "  -d                  : %s", this->Report_d ? "yes" : "no" );
    CDPrint( cdpCD0, "  -h                  : %s", this->Report_h ? "yes" : "no" );
    CDPrint( cdpCD0, "----------------------------" );

    // run the "XM count" application
    xmc.Main();

    // emit results
    CDPrint( cdpCD0, "" );
    CDPrint( cdpCD0, "----------------------------" );

    // emit optional MAPQ distribution
    if( this->Report_d )
    {
        CDPrint( cdpCD0, "MAPQ distribution:" );

        // find the maximum MAPQ count
        INT64 maxCount = 0;
        for( size_t n=0; n<arraysize(this->SFPI.MapqCount); ++n )
            maxCount = max2(maxCount, this->SFPI.MapqCount[n]);

        // emit nonzero MAPQ values and corresponding counts
        if( maxCount )
        {
            INT32 cch = static_cast<INT32>(log10(static_cast<float>(maxCount))) + 1;

            for( size_t n=0; n<arraysize(this->SFPI.MapqCount); ++n )
            {
                if( this->SFPI.MapqCount[n] )
                    CDPrint( cdpCD0, " %2d %*lld", n, cch, this->SFPI.MapqCount[n] );
            }
        }
        else
            CDPrint( cdpCD0, " (no MAPQ values counted)" );
                
        CDPrint( cdpCD0, "" );
    }

    // compute totals
    INT64 totalCm = this->SFPI.Cm_CpG + this->SFPI.Cm_CHG + this->SFPI.Cm_CHH + this->SFPI.Cm_unknown; 
    INT64 totalC = this->SFPI.C_CpG + this->SFPI.C_CHG + this->SFPI.C_CHH + this->SFPI.C_unknown;
    INT64 totalAllC = totalCm + totalC;
    INT64 totalCpG = this->SFPI.Cm_CpG + this->SFPI.C_CpG;
    INT64 totalCHG = this->SFPI.Cm_CHG + this->SFPI.C_CHG;
    INT64 totalCHH = this->SFPI.Cm_CHH + this->SFPI.C_CHH;
    INT64 totalUnk = this->SFPI.Cm_unknown + this->SFPI.C_unknown;

    // emit counts
    CDPrint( cdpCD0, "Methylation context summary:" );
    INT32 cch = totalAllC ? static_cast<INT32>(log10(static_cast<float>(totalAllC))) + 1 : 1;

    INT32 msElapsed = m_hrt.GetElapsed( false );
    CDPrint( cdpCD0, " SAM records          : %*lld", cch, this->SFPI.TotalRows );
    CDPrint( cdpCD0, " XM fields            : %*lld", cch, this->SFPI.TotalXM );
    CDPrint( cdpCD0, " Total Cs analyzed    : %*lld", cch, totalAllC );
    CDPrint( cdpCD0, "  in CpG context      : %*lld", cch, this->SFPI.Cm_CpG+this->SFPI.C_CpG );
    CDPrint( cdpCD0, "  in CHG context      : %*lld", cch, this->SFPI.Cm_CHG+this->SFPI.C_CHG );
    CDPrint( cdpCD0, "  in CHH context      : %*lld", cch, this->SFPI.Cm_CHH+this->SFPI.C_CHH );
    CDPrint( cdpCD0, "  in unknown context  : %*lld", cch, this->SFPI.Cm_unknown + this->SFPI.C_unknown );
    CDPrint( cdpCD0, " Methylated Cs        : %*lld (%4.2f%%)", cch, totalCm, 100.0*totalCm/totalAllC );
    CDPrint( cdpCD0, "  in CpG context      : %*lld (%4.2f%%)", cch, this->SFPI.Cm_CpG, 100.0*this->SFPI.Cm_CpG/totalCpG );
    CDPrint( cdpCD0, "  in CHG context      : %*lld (%4.2f%%)", cch, this->SFPI.Cm_CHG, 100.0*this->SFPI.Cm_CHG/totalCHG );
    CDPrint( cdpCD0, "  in CHH context      : %*lld (%4.2f%%)", cch, this->SFPI.Cm_CHH, 100.0*this->SFPI.Cm_CHH/totalCHH );
    CDPrint( cdpCD0, "  in unknown context  : %*lld (%4.2f%%)", cch, this->SFPI.Cm_unknown, 100.0*this->SFPI.Cm_unknown/totalUnk );
    CDPrint( cdpCD0, " Unmethylated Cs      : %*lld (%4.2f%%)", cch, totalC, 100.0*totalC/totalAllC );
    CDPrint( cdpCD0, "  in CpG context      : %*lld (%4.2f%%)", cch, this->SFPI.C_CpG, 100.0*this->SFPI.C_CpG/totalCpG );
    CDPrint( cdpCD0, "  in CHG context      : %*lld (%4.2f%%)", cch, this->SFPI.C_CHG, 100.0*this->SFPI.C_CHG/totalCHG );
    CDPrint( cdpCD0, "  in CHH context      : %*lld (%4.2f%%)", cch, this->SFPI.C_CHH, 100.0*this->SFPI.C_CHH/totalCHH);
    CDPrint( cdpCD0, "  in unknown context  : %*lld (%4.2f%%)", cch, this->SFPI.C_unknown, 100.0*this->SFPI.C_unknown/totalUnk );
    CDPrint( cdpCD0, " Elapsed              : %d.%0d (%lld rows/second)" , msElapsed/1000, msElapsed%1000, (1000*this->SFPI.TotalRows)/msElapsed );
}
#pragma endregion
