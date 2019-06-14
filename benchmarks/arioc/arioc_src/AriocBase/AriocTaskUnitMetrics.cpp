/*
   AriocTaskUnitMetrics.cpp

    Copyright (c) 2015-2019 Johns Hopkins University.  All rights reserved.

    This file is part of the Arioc software distribution.  It is subject to the license terms
    in the LICENSE.txt file found in the top-level directory of the Arioc software distribution.
    The contents of this file, in whole or in part, may only be copied, modified, propagated, or
    redistributed in accordance with the license terms contained in LICENSE.txt.
*/
#include "stdafx.h"

#pragma region static member variable definitions
const double AriocTaskUnitMetrics::m_usPerSec = 1000000;
const double AriocTaskUnitMetrics::m_msPerSec = 1000;
const double AriocTaskUnitMetrics::m_bytesPerMB = 1024*1024;
const double AriocTaskUnitMetrics::m_bytesPerGB = 1024*1024*1024;
const char* AriocTaskUnitMetrics::m_fmt_n1 = "%s%-*s: %*lld";
const char* AriocTaskUnitMetrics::m_fmt_n2 = "%s%-*s: %*lld (%2.1f%%)";
const char* AriocTaskUnitMetrics::m_fmt_n3 = "%s%-*s: %*lld (%2.1f%% of pairs)";
const char* AriocTaskUnitMetrics::m_fmt_n4 = "%s%-*s: %*u";
const char* AriocTaskUnitMetrics::m_fmt_cb1 = "%s%-*s: %*lld (%2.1fGB) (%2.1f%%)";
const char* AriocTaskUnitMetrics::m_fmt_cb2 = "%s%-*s: %*lld (%2.1fGB)";
const char* AriocTaskUnitMetrics::m_fmt_ms1 = "%s%-*s: %*lld";
#pragma endregion

#pragma region constructor/destructor
/// default constructor
AriocTaskUnitMetrics::AriocTaskUnitMetrics() : emitted(false)
{
    memset( &this->n, 0, sizeof(AriocTaskUnitMetrics::n) );
    memset( &this->cb, 0, sizeof(AriocTaskUnitMetrics::cb) );
    memset( &this->us, 0, sizeof( AriocTaskUnitMetrics::us ) );
    memset( &this->ms, 0, sizeof( AriocTaskUnitMetrics::ms ) );
    memset( &this->Key, 0, sizeof(AriocTaskUnitMetrics::Key) );
}

/// destructor
AriocTaskUnitMetrics::~AriocTaskUnitMetrics()
{
}
#pragma endregion

#pragma region private methods
/// [private] method isNull
bool AriocTaskUnitMetrics::isNull()
{
    // treat the counter structs as a sequence of 32-bit words and look for nonzero values
    UINT32* p = reinterpret_cast<UINT32*>(&this->n);
    UINT32* pLimit = reinterpret_cast<UINT32*>(&this->Key);
    while( p < pLimit )
    {
        if( *p )
            return false;

        p++ ;
    }

    return true;
}

/// [private] method usToMs
UINT64 AriocTaskUnitMetrics::usToMs( const UINT64 us )
{
    // convert microseconds to milliseconds with rounding to the nearest millisecond
    return (us + 500) / 1000;
}
#pragma endregion

#pragma region public methods
/// <summary>
/// Displays or prints the non-null contents of the current <c>AriocTaskUnitMetrics</c> instance.
/// </summary>
/// <param name="cdp">flags for <c>CDPrint</c></param>
/// <param name="flags">specifies which metric(s) are emitted even when zero</param>
/// <param name="description">description header</param>
/// <param name="cchIndent1">level 1 indentation (for description header only)</param>
/// <param name="cchIndent2">level 2 indentation (for labels; includes level 1 indentation)</param>
/// <param name="cchSpan1">column width for labels</param>
/// <param name="cchSpan2">[optional] column width for right-aligned numerical values</param>
void AriocTaskUnitMetrics::Print( CDPrintFlags cdp, const char* description, TumFlags flags, INT32 cchIndent1, INT32 cchIndent2, INT32 cchSpan1, INT32 cchSpan2 )
{
    // set up two strings of spaces to serve as indentation
    WinGlobalPtr<char> indent1(cchIndent1+1, false);
    memset( indent1.p, ' ', cchIndent1 );
    indent1.p[cchIndent1] = 0;

    cchIndent2 += cchIndent1;       // (second-level indentation is relative to the first-level indentation)
    WinGlobalPtr<char> indent2(cchIndent2+1, false);
    memset( indent2.p, ' ', cchIndent2 );
    indent2.p[cchIndent2] = 0;

    // use the key (if any) and description (if any) as a header
    if( description )
    {
        if( this->Key[0] )
            CDPrint( cdp, "%s[%s] %s:", indent1.p, this->Key, description );
        else
            CDPrint( cdp, "%s%s:", indent1.p, description );
    }
    else
    {
        if( this->Key[0] )
            CDPrint( cdp, "%s[%s]", indent1.p, this->Key );
    }

    if( this->isNull() )
    {
        // assume that elapsed time is the counter of interest
        CDPrint( cdp, "%s%-*s: <1", indent2.p, cchSpan1, "elapsed", cchSpan2 );
        return;
    }

    // initialize
    if( this->n.CandidateQ && (flags & tumMappedPairs) )
    {
        if( this->n.CandidateP == 0 )
            this->n.CandidateP = this->n.CandidateQ / 2;
    }

    // compute the column width for numerical data
    if( cchSpan2 == -1 )
    {
        // find the maximum numerical value
        UINT64 maxval = 0;
        if( this->n.CandidateQ > maxval )       maxval = this->n.CandidateQ;
        if( this->n.CandidateD > maxval )       maxval = this->n.CandidateD;
        if( this->n.CandidateP > maxval )       maxval = this->n.CandidateP;
        if( this->n.MappedQ > maxval)           maxval = this->n.MappedQ;
        if( this->n.MappedD > maxval)           maxval = this->n.MappedD;
        if( this->n.MappedP > maxval)           maxval = this->n.MappedP;
        if( this->n.Instances > maxval)         maxval = this->n.Instances;
        if( this->n.Iterations> maxval)         maxval = this->n.Iterations;
        if( this->cb.MaxAvailable > maxval )    maxval = this->cb.MaxAvailable;
        if( this->cb.MaxUsed > maxval )         maxval = this->cb.MaxUsed;
        if( this->cb.TotalBytes > maxval )      maxval = this->cb.TotalBytes;
        if( this->cb.TotalRAM > maxval )        maxval = this->cb.TotalRAM;
        if( this->us.Elapsed > maxval )         maxval = this->us.Elapsed;
        if( this->us.PreLaunch > maxval )       maxval = this->us.PreLaunch;
        if( this->us.Launch > maxval )          maxval = this->us.Launch;
        if( this->us.PostLaunch > maxval )      maxval = this->us.PostLaunch;
        if( this->ms.Elapsed > maxval )         maxval = this->ms.Elapsed;
        if( this->ms.PreLaunch > maxval)        maxval = this->ms.PreLaunch;
        if( this->ms.Launch > maxval)           maxval = this->ms.Launch;
        if( this->ms.PostLaunch > maxval)       maxval = this->ms.PostLaunch;

        if( flags & tumMappedPairs )
        {
            for( INT16 uv=0; uv<9; ++uv )
            {
                if( this->n.MappedPairs[uv] > maxval )
                    maxval = this->n.MappedPairs[uv];
            }
        }

        cchSpan2 = maxval ? static_cast<INT32>(log10( static_cast<double>(maxval) )) + 1 : 1;
    }

    // emit microsecond timings
    if( this->us.Elapsed || (flags & tumElapsed) )      CDPrint( cdp, m_fmt_ms1, indent2.p, cchSpan1, "elapsed", cchSpan2, usToMs(this->us.Elapsed) );
    if( this->us.PreLaunch || (flags & tumPreLaunch) )  CDPrint( cdp, m_fmt_ms1, indent2.p, cchSpan1, "pre-launch", cchSpan2, usToMs(this->us.PreLaunch) );
    if( this->us.Launch || (flags & tumLaunch) )
    {
        UINT64 nCandidates = (flags & tumCandidatesPerSecond) ? max2(this->n.CandidateQ,this->n.CandidateD) : 0;
        if( nCandidates )
            CDPrint( cdp, "%s%-*s: %*lld (%1.0f candidates/second)", indent2.p, cchSpan1, "kernel", cchSpan2, usToMs(this->us.Launch), m_usPerSec*nCandidates/this->us.Launch );
        else
            CDPrint( cdp, m_fmt_ms1, indent2.p, cchSpan1, "kernel", cchSpan2, usToMs(this->us.Launch) );
    }
    if( this->us.PostLaunch || (flags & tumPostLaunch) )    CDPrint( cdp, m_fmt_ms1, indent2.p, cchSpan1, "post-launch", cchSpan2, usToMs(this->us.PostLaunch) );

    // emit millisecond timings
    if( this->ms.Elapsed || (flags & tumElapsed) )      CDPrint( cdp, m_fmt_ms1, indent2.p, cchSpan1, "elapsed", cchSpan2, this->ms.Elapsed );
    if( this->ms.PreLaunch || (flags & tumPreLaunch) )  CDPrint( cdp, m_fmt_ms1, indent2.p, cchSpan1, "pre-launch", cchSpan2, this->ms.PreLaunch );
    if( this->ms.Launch || (flags & tumLaunch) )
    {
        UINT64 nCandidates = (flags & tumCandidatesPerSecond) ? max2( this->n.CandidateQ, this->n.CandidateD ) : 0;
        if( nCandidates )
            CDPrint( cdp, "%s%-*s: %*lld (%1.0f candidates/second)", indent2.p, cchSpan1, "kernel", cchSpan2, this->ms.Launch, m_msPerSec*nCandidates/this->ms.Launch );
        else
            CDPrint( cdp, m_fmt_ms1, indent2.p, cchSpan1, "kernel", cchSpan2, this->ms.Launch );
    }
    if( this->ms.PostLaunch || (flags & tumPostLaunch) )    CDPrint( cdp, m_fmt_ms1, indent2.p, cchSpan1, "post-launch", cchSpan2, this->ms.PostLaunch );

    // emit formatted numerical values
    if( this->n.CandidateQ || (flags & tumCandidateQ) ) CDPrint( cdp, m_fmt_n1, indent2.p, cchSpan1, "candidate Q sequences", cchSpan2, this->n.CandidateQ );
    if( this->n.CandidateD || (flags & tumCandidateD) ) CDPrint( cdp, m_fmt_n1, indent2.p, cchSpan1, "candidate D values", cchSpan2, this->n.CandidateD );
    if( this->n.CandidateP || (flags & tumCandidateP) ) CDPrint( cdp, m_fmt_n1, indent2.p, cchSpan1, "candidate pairs", cchSpan2, this->n.CandidateP );

    if( this->n.MappedQ || (flags & tumMappedQ) )       CDPrint( cdp, m_fmt_n2, indent2.p, cchSpan1, "mapped Q sequences", cchSpan2, this->n.MappedQ, 100.0*this->n.MappedQ/this->n.CandidateQ );
    if( this->n.MappedD || (flags & tumMappedD) )       CDPrint( cdp, m_fmt_n2, indent2.p, cchSpan1, "mapped D values", cchSpan2, this->n.MappedD, 100.0*this->n.MappedD/this->n.CandidateD );
    if( this->n.MappedP || (flags & tumMappedP) )       CDPrint( cdp, m_fmt_n2, indent2.p, cchSpan1, "paired D values", cchSpan2, this->n.MappedP, 100.0*this->n.MappedP/this->n.CandidateD );

    if( this->n.Instances || (flags & tumInstances) )
        CDPrint( cdp, m_fmt_n4, indent2.p, cchSpan1, "instances", cchSpan2, this->n.Instances );
    if( this->n.Iterations || (flags & tumIterations) )
    {
        char buf[256];
        int cb = sprintf_s( buf, sizeof buf, m_fmt_n4, indent2.p, cchSpan1, "iterations", cchSpan2, this->n.Iterations );

        if( this->n.Instances )
            cb += sprintf_s( buf+cb, (sizeof buf)-cb, " (%2.1f iterations/instance)", static_cast<double>(this->n.Iterations)/this->n.Instances );
        if( this->n.CandidateD )
            sprintf_s( buf+cb, (sizeof buf)-cb, " (%1.0f candidates/iteration)", static_cast<double>(this->n.CandidateD)/this->n.Iterations);

        CDPrint( cdp, buf );
    }

    if( this->cb.MaxAvailable || (flags & tumMaxAvailable) )
    {
        const char* banner = "bytes available";
        if( this->cb.MaxAvailablePct )
            CDPrint( cdp, m_fmt_cb1, indent2.p, cchSpan1, banner, cchSpan2, this->cb.MaxAvailable, this->cb.MaxAvailable/m_bytesPerGB, this->cb.MaxAvailablePct );
        else
            CDPrint( cdp, m_fmt_cb2, indent2.p, cchSpan1, banner, cchSpan2, this->cb.MaxAvailable, this->cb.MaxAvailable/m_bytesPerGB );
    }
    if( this->cb.MaxUsed || (flags & tumMaxUsed) )
    {
        const char* banner = "bytes used";
        if( this->cb.MaxUsedPct )
            CDPrint( cdp, m_fmt_cb1, indent2.p, cchSpan1, banner, cchSpan2, this->cb.MaxUsed, this->cb.MaxUsed/m_bytesPerGB, this->cb.MaxUsedPct );
        else
            CDPrint( cdp, m_fmt_cb2, indent2.p, cchSpan1, banner, cchSpan2, this->cb.MaxUsed, this->cb.MaxUsed/m_bytesPerGB );
    }
    if( this->cb.TotalBytes || (flags & tumTotalBytes) )
        CDPrint( cdp, m_fmt_cb2, indent2.p, cchSpan1, "total bytes", cchSpan2, this->cb.TotalBytes, this->cb.TotalBytes/m_bytesPerGB );

    // emit special-case metrics
    if( this->n.CandidateQ && (flags & tumMappedPairs) )
    {
        for( INT16 u=0; u<=2; ++u )
        {
            for( INT16 v=0; v<=2; ++v )
            {
                char banner[8];
                sprintf_s( banner, sizeof banner, "%dp%d", u, v );
                CDPrint( cdpCD2, m_fmt_n2, indent2.p, cchSpan1, banner, cchSpan2, this->n.MappedPairs[u*3+v], 100.0*this->n.MappedPairs[u*3+v]/this->n.CandidateP );
            }
        }

        UINT64 nMapped1Pairs = this->n.MappedPairs[1] + this->n.MappedPairs[2] + this->n.MappedPairs[3] + this->n.MappedPairs[6];
        UINT64 nMapped2Pairs = this->n.MappedPairs[4] + this->n.MappedPairs[5] + this->n.MappedPairs[7] + this->n.MappedPairs[8];
        CDPrint( cdpCD2, m_fmt_n3, indent2.p, cchSpan1, "pairs with two mapped mates", cchSpan2, nMapped2Pairs, 100.0*nMapped2Pairs/this->n.CandidateP );
        CDPrint( cdpCD2, m_fmt_n3, indent2.p, cchSpan1, "pairs with one mapped mate",  cchSpan2, nMapped1Pairs, 100.0*nMapped1Pairs/this->n.CandidateP );
    }

    if( this->n.MaxQperBatch )      CDPrint( cdp, m_fmt_n2, indent2.p, cchSpan1, "maximum Q sequences per batch", cchSpan2, this->n.MaxQperBatch );
    if( this->n.Batches)            CDPrint( cdp, m_fmt_n4, indent2.p, cchSpan1, "batches completed", cchSpan2, this->n.Batches );
    if( this->n.BatchInstances)     CDPrint( cdp, m_fmt_n4, indent2.p, cchSpan1, "batch instances", cchSpan2, this->n.BatchInstances);
    if( this->n.GPUs)               CDPrint( cdp, m_fmt_n4, indent2.p, cchSpan1, "GPUs", cchSpan2, this->n.GPUs );
    if( this->n.CPUthreads)         CDPrint( cdp, m_fmt_n4, indent2.p, cchSpan1, "CPU threads", cchSpan2, this->n.CPUthreads );
    if( this->cb.TotalRAM )         CDPrint( cdp, m_fmt_cb2, indent2.p, cchSpan1, "physical memory", cchSpan2, this->cb.TotalRAM, this->cb.TotalRAM/m_bytesPerGB );

    // set a flag
    this->emitted = true;
}

/// <summary>
/// Accumulates the specified maximum value.
/// </summary>
/// <param name="flag">indicates which counter to accumulate</param>
/// <param name="value">value to accumulate</param>
void AriocTaskUnitMetrics::AccumulateMax( TumFlags flag, UINT64 value )
{
    UINT64* pm = NULL;
    switch( flag )
    {
        case tumMaxAvailable:
            pm = &this->cb.MaxAvailable;
            break;

        case tumMaxUsed:
            pm = &this->cb.MaxUsed;
            break;

        default:
            throw new ApplicationException( __FILE__, __LINE__, "invalid flag" );
    }

    // we need a critical section here because there is no "atomic max" function
    {
        RaiiCriticalSection<AriocTaskUnitMetrics> rcs;

        if( value > *pm )
            *pm = value;
    }
}

/// <summary>
/// Accumulates the specified maximum value.
/// </summary>
/// <param name="flag">indicates which counter to accumulate</param>
/// <param name="value">value to accumulate</param>
void AriocTaskUnitMetrics::AccumulateMax( TumFlags flag, double value )
{
    double* pm = NULL;
    switch( flag )
    {
        case tumMaxAvailablePct:
            pm = &this->cb.MaxAvailablePct;
            break;

        case tumMaxUsedPct:
            pm = &this->cb.MaxUsedPct;
            break;

        default:
            throw new ApplicationException( __FILE__, __LINE__, "invalid flag" );
    }

    // we need a critical section here because there is no "atomic max" function
    {
        RaiiCriticalSection<AriocTaskUnitMetrics> rcs;

        if( value > *pm )
            *pm = value;
    }
}

/// <summary>
/// Accumulates the specified maximum value and percent capacity.
/// </summary>
/// <param name="flag">indicates which pair of counters to accumulate</param>
/// <param name="value">value to accumulate</param>
/// <param name="capacity">capacity (maximum value)</param>
void AriocTaskUnitMetrics::AccumulateMaxPct( TumFlags flag, UINT64 value, UINT64 capacity )
{
    // compute the percentage
    double pct = static_cast<double>(100.0 * value / capacity);

    UINT64* pm = NULL;
    double* ppct = NULL;
    switch( flag )
    {
        case tumMaxAvailable:
            pm = &this->cb.MaxAvailable;
            ppct = &this->cb.MaxAvailablePct;
            break;

        case tumMaxUsed:
            pm = &this->cb.MaxUsed;
            ppct = &this->cb.MaxUsedPct;
            break;

        default:
            throw new ApplicationException( __FILE__, __LINE__, "invalid flag" );
    }

    // we need a critical section here because there is no "atomic max" function
    {
        RaiiCriticalSection<AriocTaskUnitMetrics> rcs;

        if( pct > *ppct )
        {
            *pm = value;
            *ppct = pct;
        }
    }
}
#pragma endregion
