/*
  baseMapCommon.h

    Copyright (c) 2015-2019 Johns Hopkins University.  All rights reserved.

    This file is part of the Arioc software distribution.  It is subject to the license terms
    in the LICENSE.txt file found in the top-level directory of the Arioc software distribution.
    The contents of this file, in whole or in part, may only be copied, modified, propagated, or
    redistributed in accordance with the license terms contained in LICENSE.txt.
*/
#pragma once
#define __baseMapCommon__

/// <summary>
/// Class <c>baseMapCommon</c> implements code used by the "baseMap" implementations in both AriocU and AriocP.
/// </summary>
class baseMapCommon : public CudaLaunchCommon
{
    protected:
        QBatch*                 m_pqb;
        AriocBase*              m_pab;
        DeviceBuffersG*         m_pdbg;
        HostBuffers*            m_phb;
        WinGlobalPtr<UINT32>    m_cnJ;
        UINT32                  m_isi;          // seed iteration loop index
        UINT32                  m_isiLimit;     // seed iteration loop limit
        AriocTaskUnitMetrics*   m_ptum;
        HiResTimer              m_hrt;

    private:
        UINT32 nJforQIDrange( UINT32 ofs0, UINT32 ofs1 );

    protected:
        baseMapCommon( void );
        baseMapCommon( QBatch* pqb, DeviceBuffersG* pdbg, HostBuffers* phb );
        ~baseMapCommon( void );
        void exciseRedundantDvalues( DeviceBuffersG* pdbg );
        void prepareIteration( const UINT32 nQremaining, const INT64 nJremaining, UINT32& iQ, UINT32& nQ, UINT32& nJ );
};
