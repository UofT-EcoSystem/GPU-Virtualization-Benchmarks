/*
  AriocBase.cpp

    Copyright (c) 2015-2019 Johns Hopkins University.  All rights reserved.

    This file is part of the Arioc software distribution.  It is subject to the license terms
    in the LICENSE.txt file found in the top-level directory of the Arioc software distribution.
    The contents of this file, in whole or in part, may only be copied, modified, propagated, or
    redistributed in accordance with the license terms contained in LICENSE.txt.
*/
#include "stdafx.h"

#pragma region static member variables
AriocAppMetrics           AriocBase::aam;
AA<AriocTaskUnitMetrics>  AriocBase::tum(128);      // 128: large enough so that the associative array won't be reallocated
#pragma endregion

#pragma region constructors and destructor
/// [public] constructor
AriocBase::AriocBase( const char* _pathR, A21SpacedSeed _a21ss, A21HashedSeed _a21hs, AlignmentControlParameters _acp, AlignmentScoreParameters _asp,
                      InputFileGroup* _pifgQ,
                      WinGlobalPtr<OutputFileInfo>& _ofi,
                      UINT32 _gpuMask, INT32 _maxDOP, UINT32 _batchSize, INT32 _kmerSize, CIGARfmtType _cigarFmtType, MDfmtType _mdFmtType,
                      AriocAppMainBase* _paamb ) : m_cbTotalAvailable(0), m_cbTotalUsed(0), m_cbTotalPinned(0),
                                                   m_rdiNongapped(_pathR,_a21ss), m_rdiGapped(_pathR,_a21hs),
                                                   nLPs(0), nGPUs(0), nReadsInFiles(0), nReadsAligned(0),
                                                   RefId(0), minSubId(0), maxSubId(0), BatchSize(_batchSize),
                                                   R(NULL), celR(0), usePinnedR(false),
                                                   pifgQ(_pifgQ),
                                                   a21ss(_a21ss), a21hs(_a21hs), aas(_acp, _asp),
                                                   paamb(_paamb), SAMhdb(&_paamb->RGMgr),
                                                   doMainLoopJoin(false), preferLoadRix(true), TLENbias(0),
                                                   nTLEN(0), sumTLEN(0), sosTLEN(0), iMeanTLEN(0), dMeanTLEN(0), stdevTLEN(0),
                                                   KmerSize(_kmerSize), CFT(_cigarFmtType), MFT(_mdFmtType),
                                                   StrandsPerSeed((_a21ss.baseConvert!=A21SeedBase::bcNone) ? 2 : 1),
                                                   Watchdog(this)
{
    sniffHostResources( _maxDOP );
    sniffGpuResources( _gpuMask );
    initARowWriters( _ofi );

    // load read-group info for the input files
    paamb->RGMgr.LoadReadGroupInfo( _pifgQ );

    // conditionally get SAM info from the .cfg files associated with the spaced-seed and seed-and-extend lookup tables
    if( this->SAMwriter.n )
        SAMhdb.Init( m_rdiNongapped.rfiJ.Dir.DirSpec, m_rdiGapped.rfiJ.Dir.DirSpec, _pifgQ, _paamb );

    // conditionally write read-group info
    if( this->TSEwriter.n )
        paamb->RGMgr.WriteReadGroupInfo( _ofi );

    // save a lookup table of read-group ID strings
    paamb->RGMgr.SaveRGIDs();

    // look for the Xparam that specifies whether to synchronize the main loop in the GPU-associated CPU threads
    INT32 i = _paamb->Xparam.IndexOf( "wantMainLoopJoin" );
    if( i >= 0 )
        this->doMainLoopJoin = (_paamb->Xparam.Value(i) != 0);

    // look for the Xparam that specifies the worker thread launch timeout
    i = _paamb->Xparam.IndexOf( "workerThreadLaunchTimeout" );
    if( i >= 0 )
        tuBaseA::WorkerThreadLaunchTimeout = static_cast<DWORD>(_paamb->Xparam.Value(i));

    // look for the Xparam that specifies the maximum size of a persistent Windows global memory allocation
    i = _paamb->Xparam.IndexOf( "cbmaxWinGlobalAlloc" );
    if( i >= 0 )
        WinGlobalPtrHelper::cbmaxWinGlobalAlloc = static_cast<size_t>(_paamb->Xparam.Value( i ));

    // look for the Xparam that specifies "optimized" loading of interleaved R sequence data
    i = _paamb->Xparam.IndexOf( "loadRix" );
    if( i >= 0 )
        this->preferLoadRix = (_paamb->Xparam.Value(i) != 0);

    // look for the optional "TLENbias" Xparam
    i = _paamb->Xparam.IndexOf( "TLENbias" );
    if( i >= 0 )
        this->TLENbias = static_cast<INT32>(_paamb->Xparam.Value( i ));
}

/// [public] destructor
AriocBase::~AriocBase()
{
    zapARowWriters( this->SAMwriter );
    zapARowWriters( this->SBFwriter );
    zapARowWriters( this->TSEwriter );
    zapARowWriters( this->KMHwriter );
}
#pragma endregion

#pragma region private methods
/// [private static] method fileSubIdComparer
int AriocBase::fileSubIdComparer( const void* a, const void* b )
{
    const short3* pa = reinterpret_cast<const short3*>(a);
    const short3* pb = reinterpret_cast<const short3*>(b);

    // order by y (pair flag), then x (adjusted subId)
    int rval = pa->y - pb->y;
    if( rval == 0 )
        rval = pa->x - pb->x;
    return rval;
}

/// [private] method initARowWriters
void AriocBase::initARowWriters( WinGlobalPtr<OutputFileInfo>& ofi )
{
    /* Allocate one baseARowWriter instance (that is, one per potential output file) for each possible combination of alignment-result reporting flags.

        For paired-end reads, there are four mutually-exclusive 1-bit flags:
            arfReportConcordant
            arfReportDiscordant
            arfReportRejected
            arfReportUnmapped

        For unpaired reads, there are two mutually-exclusive 1-bit flags:
            arfMapped
            arfUnmapped

        In both cases, we build a list of ARowWriter instances such that a flag value serves as an offset into the list, so that the correct ARowWriter
         instance (corresponding to the file where reads with the user-specified flag(s) are written) is accessed using syntax like the following:

            char* pbuf = pArioc->SBFwriter.p[arfMapped]->LockOutputBuffer( ... );

        Since we're using bitmask values as list indices, the only elements in the list that can be used are powers of two.  (We trade a bit of space
        in the list of ARowWriter instances for a simpler array subscript.)
    */
    
    // allocate and zero a buffer for each possible output file format
    INT16 cel = (this->pifgQ->HasPairs ? arfMaxReportPaired : arfMaxReportUnpaired) + 1;
    this->SAMwriter.Realloc( cel, true );
    this->SBFwriter.Realloc( cel, true );
    this->TSEwriter.Realloc( cel, true );
    this->KMHwriter.Realloc( cel, true );

    // traverse the list of output files
    for( size_t n=0; n<ofi.Count; ++n )
    {
        /* Set up for the nth output file path:
            - if the nth file specification is non-null, instantiate a new ARowWriter and use it for each of the alignment result flags associated with
                the output file format
            - otherwise, use a null ARowWriter reference as a placeholder; a baseARowWriter instance will be used instead
        */
        ARowWriter* pAriocaw = (*ofi.p[n].path) ? new ARowWriter( this, ofi.p+n ) : NULL;

        // get the first flag for the output file
        UINT16 iarf = ofi.p[n].arf;
        DWORD pos;                           // 0-based position of the low-order 1 bit
        while( _BitScanForward( &pos, iarf ) )
        {
            // build the bit mask that corresponds to the flag at the current position in the alignment result flags
            AlignmentResultFlags arf = static_cast<AlignmentResultFlags>(1 << pos);

            switch( ofi.p[n].oft )
            {
                case oftSAM:
                    if( this->SAMwriter.p[arf] != NULL )
                        throw new ApplicationException( __FILE__, __LINE__, "redundantly defined SAM output: %s", AriocAlignmentScorer::AlignmentResultFlagsToString( arf, arfMaskReport ) );
                    this->SAMwriter.p[arf] = pAriocaw;
                    this->SAMwriter.n++ ;
                    break;

                case oftSBF:
                    if( this->SBFwriter.p[arf] != NULL )
                        throw new ApplicationException( __FILE__, __LINE__, "redundantly defined SBF output: %s", AriocAlignmentScorer::AlignmentResultFlagsToString( arf, arfMaskReport ) );
                    this->SBFwriter.p[arf] = pAriocaw;
                    this->SBFwriter.n++;
                    break;

                case oftTSE:
                    if( this->TSEwriter.p[arf] != NULL )
                        throw new ApplicationException( __FILE__, __LINE__, "redundantly defined TSE output: %s", AriocAlignmentScorer::AlignmentResultFlagsToString( arf, arfMaskReport ) );
                    this->TSEwriter.p[arf] = pAriocaw;
                    this->TSEwriter.n++;
                    break;

                case oftKMH:
                    if( this->KMHwriter.p[arf] != NULL )
                        throw new ApplicationException( __FILE__, __LINE__, "redundantly defined KMH output: %s", AriocAlignmentScorer::AlignmentResultFlagsToString( arf, arfMaskReport ) );
                    this->KMHwriter.p[arf] = pAriocaw;
                    this->KMHwriter.n++;
                    break;

                default:
                    throw new ApplicationException( __FILE__, __LINE__, "unrecognized OutputFileType: %d", ofi.p[n].oft );
            }

            // move to the next bit in the alignment result flags
            iarf ^= arf;
        }
    }

    // for alignment result flags without an ARowWriter, use the baseARowWriter implementation
    INT16 u = 0;
    do
    {
        // ensure that the nth element in the list is initialized
        if( this->SAMwriter.p[u] == NULL )
            this->SAMwriter.p[u] = new baseARowWriter( this );
        if( this->SBFwriter.p[u] == NULL )
            this->SBFwriter.p[u] = new baseARowWriter( this );
        if( this->TSEwriter.p[u] == NULL )
            this->TSEwriter.p[u] = new baseARowWriter( this );
        if( this->KMHwriter.p[u] == NULL )
            this->KMHwriter.p[u] = new baseARowWriter( this );

        // advance to the next power of two
        u = (u ? (u << 1) : 1);
    }
    while( u < cel );
}

/// [private] method zapARowWriters
void AriocBase::zapARowWriters( WinGlobalPtr<baseARowWriter*>& arwList )
{
    // traverse the list of ARowWriter instances
    for( INT16 u=0; u<static_cast<INT16>(arwList.Count); ++u )
    {
        // look for non-null elements in the list
        if( arwList.p[u] )
        {
            // delete the uth ARowWriter instance in the list
            delete arwList.p[u];

            // null subsequent list entries that referenced the deleted instance
            for( INT16 v=u+1; v<static_cast<INT16>(arwList.Count); ++v )
            {
                if( arwList.p[v] == arwList.p[u] )
                    arwList.p[v] = NULL;
            }

            // null the uth list entry
            arwList.p[u] = NULL;
        }
    }
}

/// [private] method sniffHostResources
void AriocBase::sniffHostResources( INT32 maxDOP )
{
    // ensure we're running a 64-bit OS
    if( sizeof(void*) != 8 )
        throw new ApplicationException( __FILE__, __LINE__, "This application runs only on a 64-bit operating system." );

    // ensure that sizeof(Qwarp) is a multiple of 8
    if( sizeof(Qwarp) & 7 )
        throw new ApplicationException( __FILE__, __LINE__, "sizeof(Qwarp) must be a multiple of 8 (actual size = %d bytes)", sizeof(Qwarp) );

    // ensure that sizeof(BRLEAheader) is a multiple of 4
    if( sizeof(BRLEAheader) & 3 )
        throw new ApplicationException( __FILE__, __LINE__, "sizeof(BRLEAheader) must be a multiple of 8 (actual size = %d bytes)", sizeof(Qwarp) );

    // get the machine name
    UINT32 cbMachineName = sizeof this->paamb->MachineName;
    GetComputerName( this->paamb->MachineName, reinterpret_cast<LPDWORD>(&cbMachineName) );

    // get the number of "logical processors" (i.e. concurrent CPU threads, including hyperthreads)
    INT32 nLPs = GetAvailableCpuThreadCount();
    this->nLPs = min2(nLPs, maxDOP);       // clamp to the user-specified maximum

    // get the total amount of host memory immediately available to this process
    m_cbTotalAvailable = static_cast<INT64>(GetTotalSystemRAM());

    // performance metrics
    AriocBase::aam.n.CPUThreads = this->nLPs;
    AriocBase::aam.cb.Available = m_cbTotalAvailable;
}

/// [private] method sniffGpuResources
void AriocBase::sniffGpuResources( UINT32 gpuMask )
{
    HiResTimer hrt;

    // filter the set of available CUDA devices
    CudaDeviceBinding::DeviceFilter_MinComputeCapability( CUDAMINCC );
    CudaDeviceBinding::DeviceFilter_DeviceIdMap( gpuMask );

    // do first-time CUDA initialization on the available devices
    CudaDeviceBinding::InitializeAvailableDevices();

    // get the number of usable CUDA devices
    this->nGPUs = CudaDeviceBinding::GetAvailableDeviceCount();

    // sanity check
    if( this->nGPUs == 0 )
        throw new ApplicationException( __FILE__, __LINE__, "no GPUs with minimum compute capability %s and gpuMask=0x%08x", CUDAMINCC, gpuMask );

    // performance metrics
    AriocBase::aam.n.GPUs = this->nGPUs;
    AriocBase::aam.ms.InitializeGPUs = hrt.GetElapsed(false);
}

/// [private] method getSqIdFromFile
INT64 AriocBase::getSqIdFromFile( char* fileSpec )
{
    // open the specified file and read the first sqId (see getRfromFile for file format)
    RaiiFile rf( fileSpec, true );      // true: readonly

    INT64 sqId;
    rf.Read( &sqId, sizeof sqId );

    // return the sqId
    return sqId;
}

/// [private] method getSqIdAndMfromFile
INT64 AriocBase::getSqIdAndMfromFile( char* fileSpec )
{
    // open the specified file and read the first sqId (see AriocBase::getRfromFile for file format)
    RaiiFile rf( fileSpec, true );      // true: readonly

    INT64 sqId;
    rf.Read( &sqId, sizeof sqId );

    INT64 M;
    rf.Read( &M, sizeof M );

    // save the M value for the forward strand
    if( (sqId & AriocDS::SqId::MaskMateId) == 0 )
    {
        // get the subunit ID
        INT32 subId = static_cast<INT32>(AriocDS::SqId::GetReadId( sqId )); // the read ID of the reference sequence is the subunit ID (e.g. chromosome number)

        // sanity checks
        if( M > _I32_MAX )
            throw new ApplicationException( __FILE__, __LINE__, "file %s contains %lld symbols (maximum supported reference sequence size is %d symbols)", fileSpec, M, _I32_MAX );
        if( this->M.p[subId] != 0 )
            throw new ApplicationException( __FILE__, __LINE__, "duplicate reference-sequence subId: %d", subId );

        // save the number of symbols associated with the subunit ID
        this->M.p[subId] = static_cast<UINT32>(M);

        // performance metrics
        CDPrint( cdpCD2, "%s: subId %d: loaded %lld symbols from %s", __FUNCTION__, subId, M, rf.FileSpec.p );
    }

    // return the subunit ID
    return sqId;
}

/// [private] method getRfromFile
INT64 AriocBase::getRfromFile( INT64 ofs, char* fileSpec )
{
    // open the file
    RaiiFile rf( fileSpec, true );  // true: readonly

    /* The file is a SQL bulk-formatted row that looks like this:
        byte 0-7       :  sequence ID
        byte 8-0x0F    :  number of symbols in the sequence
        byte 0x10-0x17 :  number of bytes in the binary string
        byte 0x18-     :  binary string (representing a vector of bigint (64-bit) values)
    */

    // get the string length
    INT64 cb;
    rf.Seek( 0x10, SEEK_SET );
    rf.Read( &cb, sizeof cb );

    // copy the contents of the specified file to the current offset in the R buffer
    INT64 cbRead = rf.Read( this->R+ofs, cb );
    if( cbRead != cb )
        throw new ApplicationException( __FILE__, __LINE__, "read %lld/%lld bytes from %s", cbRead, cb, fileSpec );

    // update the offset
    return ofs + blockdiv( cbRead, sizeof(UINT64) );
}

/// [private] void appendRpadding
INT64 AriocBase::appendRpadding( INT64 ofs )
{
    UINT64* p = this->R + ofs;
    for( UINT32 n=0; n<celPad; ++n )
        *(p++) = MASK001;   // all 21 symbols are binary 001 (nonzero null symbol for a reference sequence, AKA Nr)

    return ofs + celPad;
}
#pragma endregion

#pragma region protected methods
/// [protected] method updateProgress
void AriocBase::updateProgress( UINT32 nQ )
{
    // accumulate the specified number of reads (Q sequences)
    INT64 nra0 = InterlockedExchangeAdd64( &this->nReadsAligned, nQ );

    if( CDPrintFilter & cdpCD1 )
    {
        // if the current percentage of reads aligned has crossed a "10 percent" boundary, emit a progress notification
        INT32 pa = static_cast<INT32>((100 * this->nReadsAligned) / this->nReadsInFiles);
        INT32 pa0 = static_cast<INT32>((100 * nra0) / this->nReadsInFiles);
        if( (nQ == 0) || ((pa/10) > (pa0/10)) )
        {
            // estimate the execution time remaining
            char etr[32];
            INT32 pr = 100 - pa;
            if( pa && pr )
            {
                INT32 msElapsed = m_hrt.GetElapsed( false );
                INT32 msRemaining = (msElapsed * pr) / pa;

                // display 1 decimal place when there are less than 5 minutes remaining
                if( msRemaining >= 5*60000 )
                    sprintf_s( etr, arraysize(etr), " (ETA %d minutes)", msRemaining/60000 );
                else
                    sprintf_s( etr, arraysize(etr), " (ETA %3.1f minutes)", msRemaining/60000.0 );
            }
            else
                etr[0] = '\0';

            // emit a string
            if( this->pifgQ->HasPairs )
                CDPrint( cdpCD1, "%s: %3d%%: %llu pairs (%llu mates) aligned%s", __FUNCTION__, pa, this->nReadsAligned/2, this->nReadsAligned, etr );
            else
                CDPrint( cdpCD1, "%s: %3d%%: %llu reads aligned%s", __FUNCTION__, pa, this->nReadsAligned, etr );
        }
    }
}

/// [protected] method loadR
void AriocBase::loadR()
{
    CRVALIDATOR;

    CDPrint( cdpCD0, "%s: GPU R load starts...", __FUNCTION__ );

    HiResTimer hrt(ms);

    /* The R buffer may reside either in page-locked system memory (CUDA "pinned" memory) or in CUDA global memory
        on each GPU device.
    */

    // look for the Xparam that specifies that the R buffer allocation should reside in pinned (page-locked) memory
    INT32 i = this->paamb->Xparam.IndexOf( "useRinGmem" );
    if( i >= 0 )
        this->usePinnedR = (this->paamb->Xparam.Value( i ) == 0);

    /* There may be either one or two R sequence files for each subId; if there are two, one must represent the forward
        sequence and the other the reverse complement sequence.

       The R sequence refId (specified in AriocE as the data-source ID (srcId)) is extracted from the SqId in the
        first input file.
    
       For each R sequence file
        - get the subunit ID (from the sqId for the first-and-only row in the file)
        - ensure that there is a file for each value in the range of subId values
        - read the data
    */
    ReferenceDataInfo* prdi = m_rdiNongapped.HasData ? &m_rdiNongapped : &m_rdiGapped;
    INT16 nFiles = static_cast<INT16>(prdi->rfiR.Dir.Filenames.Count);
    WinGlobalPtr<char> fileSpec;

    // determine the range of subId values
    this->minSubId = INT_MAX;
    this->maxSubId = INT_MIN;
    for( INT16 n=0; n<nFiles; ++n )
    {
        // get the nth subId and M (sequence length)
        prdi->rfiR.Dir.GetFileSpecification( n, fileSpec );
        INT64 sqId = getSqIdFromFile( fileSpec.p );
        INT32 subId = static_cast<INT32>(AriocDS::SqId::GetReadId( sqId ));

        // get the R sequence refId
        if( n == 0 )
            this->RefId = AriocDS::SqId::GetSrcId(sqId);

        // track the minimum and maximum values
        this->minSubId = min2( this->minSubId, subId );
        this->maxSubId = max2( this->maxSubId, subId );
    }

    /* allocate a buffer to contain the M values (reference sequence lengths); we allocate a buffer large enough
        so that a reference sequence's subId can be used as an index into the list of M values */
    M.Realloc( this->maxSubId+1, true );

    // get the subId and M values and count the number of files for each subId
    INT16 nDistinctSubId = static_cast<INT16>((this->maxSubId - this->minSubId) + 1);
    WinGlobalPtr<short3> dscSub( nFiles, false );
    memset( dscSub.p, 0xFF, dscSub.cb );                // initialize all values to 0xFFFF (-1)

    for( INT16 n=0; n<nFiles; ++n )
    {
        // get the nth sqId
        prdi->rfiR.Dir.GetFileSpecification( n, fileSpec );

        INT64 sqId = getSqIdAndMfromFile( fileSpec.p );
        INT32 subId = static_cast<INT32>(AriocDS::SqId::GetReadId( sqId ));

#if TODO_CHOP_WHEN_DEBUGGED
        CDPrint( cdpCD4, "n=%d subId=%d pairFlag=%d fileSpec=%s", n, subId, AriocDS::SqId::GetMateId(sqId), fileSpec.p );
#endif

        dscSub.p[n].x = subId;                              // subunit ID
        dscSub.p[n].y = AriocDS::SqId::GetMateId( sqId );   // 0: forward; 1: reverse complement
        dscSub.p[n].z = n;                                  // iFile
    }
        
    // sanity checks
    bool hasReverseComplement = false;
    for( INT16 n=0; n<nFiles; ++n )
    {
        // look for missing subId values
        if( dscSub.p[n].x < 0 )
            throw new ApplicationException( __FILE__, __LINE__, "missing subId value %d for R sequence files in %s", n+this->minSubId, prdi->rfiR.Dir.DirSpec );

        // set a flag if reverse complement sequences are present
        if( dscSub.p[n].y == 1 )
            hasReverseComplement = true;
    }

    if( hasReverseComplement )
    {
        for( INT16 n=0; n<nFiles; ++n )
        {
            if( dscSub.p[n].y < 0 )
                throw new ApplicationException( __FILE__, __LINE__, "missing reverse complement for subId=%d for R sequence files in %s", n+minSubId, prdi->rfiR.Dir.DirSpec );
        }
    }
    else
    {
        for( INT16 n=0; n<nFiles; ++n )
        {
            if( dscSub.p[n].y == 1 )
                throw new ApplicationException( __FILE__, __LINE__, "unexpected reverse complement for subId=%d for R sequence files in %s", n+minSubId, prdi->rfiR.Dir.DirSpec );
        }
    }

    /* Allocate a buffer to contain R sequence data.
    
       Each R sequence is padded with null symbols (Nr) so that there is 1Kb of padding before the first R sequence,
        between successive R sequences, and after the last R sequence (see readRfile()).
    */
    size_t cb = static_cast<size_t>(prdi->rfiR.TotalSize + ((celPad*sizeof(UINT64)) * (nFiles+2)));
    this->celR = blockdiv(cb,sizeof(INT64));

    UINT64* p;
    if( this->usePinnedR )
    {
        m_Rpinned.Realloc( this->celR, true );
        this->R = m_Rpinned.p;

        p = m_Rpinned.p;
        cb = m_Rpinned.cb;
    }
    else
    {
        m_Rbuf.Realloc( this->celR, true );
        this->R = m_Rbuf.p;

        p = m_Rbuf.p;
        cb = m_Rbuf.cb;
    }
    
    INT64 ofs = appendRpadding( 0 );    // insert padding before the first R sequence in the buffer

    CDPrint( cdpCD4, "%s: allocated %s buffer: %llu bytes (%3.1fGB) at 0x%016llx in %dms",
                     __FUNCTION__, (this->usePinnedR ? "pinned" : "host (non-pinned)"),
                     cb, cb/GIGADOUBLE, p, hrt.GetElapsed(false) );

    // sort the list of subId values by pair flag, subId
    qsort( dscSub.p, dscSub.Count, sizeof(short3), fileSubIdComparer );

    // load the R+ files in order by subId
    this->ofsRplus.Realloc( this->maxSubId+1, false );
    CREXEC( cudaMemset( this->ofsRplus.p, 0xFF, this->ofsRplus.cb ) );  // initialize with nulls (all bits set)
    for( INT16 n=0; n<nDistinctSubId; ++n )
    {
        prdi->rfiR.Dir.GetFileSpecification( dscSub.p[n].z, fileSpec );

#if TODO_CHOP_WHEN_DEBUGGED
        CDPrint( cdpCD4, "loading dscSub.p[%d].x=%d .y=%d .z=%d at offset %lld: %s", n, dscSub.p[n].x, dscSub.p[n].y, dscSub.p[n].z, ofs, fileSpec.p );
#endif

        // save the offset of the start of the encoded R sequence data for the nth file
        INT16 rid = dscSub.p[n].x;
        this->ofsRplus.p[rid] = ofs;        // this is the offset of the first encoded R symbol (i.e., after the padding)

        // copy the data into the buffer
        ofs = getRfromFile( ofs, fileSpec.p );
        ofs = appendRpadding( ofs );
    }

    if( hasReverseComplement )
    {
        // load the R- files in order by subId
        this->ofsRminus.Realloc( this->maxSubId+1, false );
        CREXEC( cudaMemset( this->ofsRminus.p, 0xFF, this->ofsRminus.cb ) );    // initialize with nulls (all bits set)
        for( INT16 n=nDistinctSubId; n<nFiles; ++n )
        {
            prdi->rfiR.Dir.GetFileSpecification( dscSub.p[n].z, fileSpec );
            
#if TODO_CHOP_WHEN_DEBUGGED
            CDPrint( cdpCDb, "loading dscSub.p[%d].x=%d .y=%d .z=%d at offset %lld: %s", n, dscSub.p[n].x, dscSub.p[n].y, dscSub.p[n].z, ofs, fileSpec.p );
#endif

            // save a reference to the start of the encoded R sequence data for the nth file
            INT16 rid = dscSub.p[n-nDistinctSubId].x;
            this->ofsRminus.p[rid] = ofs;   // this is the offset of the first encoded R symbol (i.e., after the padding)

            // copy the data into the buffer
            ofs = getRfromFile( ofs, fileSpec.p );
            ofs = appendRpadding( ofs );
        }
    }

    // save the actual number of 64-bit elements in the R buffer
    this->celR = static_cast<UINT32>(ofs);

    // performance metrics
    AriocBase::aam.ms.LoadR = hrt.GetElapsed(false);
    AriocBase::aam.cb.R = this->celR * sizeof(UINT64);

    CDPrint( cdpCD0, "%s: GPU R load complete in %ums", __FUNCTION__, AriocBase::aam.ms.LoadR );
}

/// [protected] method loadHJ
void AriocBase::loadHJ()
{
    CRVALIDATOR;

    CDPrint( cdpCD0, "%s: GPU LUT load starts...", __FUNCTION__ );

    HiResTimer hrt(ms);
    WinGlobalPtr<char> fileSpec;

    /* The H lookup table (hash table) may reside in either a page-locked (pinned) host memory buffer (the default)
        or in CUDA global memory on each GPU:
        
           - In pinned memory: we load the table directly into a pinned-memory buffer
           - In per-GPU memory: we load the table into a host buffer from which the table is subsequently copied

       By default, the H and J buffers are allocated and initialized concurrently.  The "serialLUTinit"
        Xparam may be used to cause each buffer to be allocated and initialized serially instead.
    */

    tuLoadLUT<UINT32>* pLoadHn = NULL;
    tuLoadLUT<UINT32>* pLoadJn = NULL;
    tuLoadLUT<UINT32>* pLoadHg = NULL;
    tuLoadLUT<UINT32>* pLoadJg = NULL;

    // look for the relevant Xparams
    bool serialLUTinit = false;
    INT32 i = this->paamb->Xparam.IndexOf( "serialLUTinit" );
    if( i >= 0 )
         serialLUTinit = (this->paamb->Xparam.Value(i) != 0);

    bool usePinnedH = true;
    i = this->paamb->Xparam.IndexOf( "useHinGmem" );
    if( i >= 0 )
         usePinnedH = (this->paamb->Xparam.Value(i) == 0);

    if( m_rdiNongapped.HasData )
    {
        // H lookup table
        m_rdiNongapped.rfiH.Dir.GetFileSpecification( 0, fileSpec );

        if( usePinnedH )
            pLoadHn = new tuLoadLUT<UINT32>( &this->Hn, fileSpec.p, 0 );    // page-locked ("pinned") host memory buffer
        else
            pLoadHn = new tuLoadLUT<UINT32>( &this->Hng, fileSpec.p, 0 );   // host memory buffer (not page-locked)

        pLoadHn->Start();
        if( serialLUTinit )
            pLoadHn->Wait( LUT_LOAD_TIMEOUT );

        // J lookup table
        m_rdiNongapped.rfiJ.Dir.GetFileSpecification( 0, fileSpec );
        pLoadJn = new tuLoadLUT<UINT32>( &this->Jn, fileSpec.p, 0 );        // page-locked ("pinned") host memory buffer
        pLoadJn->Start();
        if( serialLUTinit )
            pLoadJn->Wait( LUT_LOAD_TIMEOUT );
    }

    if( m_rdiGapped.HasData )
    {
        // H lookup table
        m_rdiGapped.rfiH.Dir.GetFileSpecification( 0, fileSpec );

        if( usePinnedH )
            pLoadHg = new tuLoadLUT<UINT32>( &this->Hg, fileSpec.p, 0 );    // page-locked ("pinned") host memory buffer
        else
            pLoadHg = new tuLoadLUT<UINT32>( &this->Hgg, fileSpec.p, 0 );   // host memory buffer (not page-locked)

        pLoadHg->Start();
        if( serialLUTinit )
            pLoadHg->Wait( LUT_LOAD_TIMEOUT );

        // J lookup table
        m_rdiGapped.rfiJ.Dir.GetFileSpecification( 0, fileSpec );
        pLoadJg = new tuLoadLUT<UINT32>( &this->Jg, fileSpec.p, 0 );        // page-locked ("pinned") host memory buffer
        pLoadJg->Start();
        if( serialLUTinit )
            pLoadJg->Wait( LUT_LOAD_TIMEOUT );
    }

    // discard each tuLoadLUT instance after its load completes
    if( pLoadHn )
    {
        if( !serialLUTinit ) pLoadHn->Wait( LUT_LOAD_TIMEOUT );
        delete pLoadHn;
    }

    if( pLoadJn )
    {
        if( !serialLUTinit ) pLoadJn->Wait( LUT_LOAD_TIMEOUT );
        delete pLoadJn;
    }

    if( pLoadHg )
    {
        if( !serialLUTinit ) pLoadHg->Wait( LUT_LOAD_TIMEOUT );
        delete pLoadHg;
    }

    if( pLoadJg )
    {
        if( !serialLUTinit ) pLoadJg->Wait( LUT_LOAD_TIMEOUT );
        delete pLoadJg;
    }

    // performance metrics
    AriocBase::aam.ms.LoadHJ = hrt.GetElapsed(false);
    AriocBase::aam.cb.Hn = this->Hn.cb;
    AriocBase::aam.cb.Hg = this->Hg.cb;
	AriocBase::aam.cb.Hng = this->Hng.cb;
	AriocBase::aam.cb.Hgg = this->Hgg.cb;
    AriocBase::aam.cb.Jn = this->Jn.cb;
    AriocBase::aam.cb.Jg = this->Jg.cb;
    AriocBase::aam.cb.Pinned += this->Hn.cb + this->Hg.cb + this->Jn.cb + this->Jg.cb;
	AriocBase::aam.cb.HJgmem += this->Hng.cb + this->Hgg.cb;
    JtableHeader* pJHn = reinterpret_cast<JtableHeader*>(this->Jn.p);
    AriocBase::aam.n.maxnJn = pJHn->maxnJ;                                // maximum J-list size
    JtableHeader* pJHg = reinterpret_cast<JtableHeader*>(this->Jg.p);
    AriocBase::aam.n.maxnJg = pJHg->maxnJ;
    if( pJHn->maxSubId != pJHg->maxSubId )
        throw new ApplicationException( __FILE__, __LINE__, "maximum subunit ID in nongapped J table: %d; in gapped J table: %d", pJHn->maxSubId, pJHg->maxSubId );
    AriocBase::aam.n.maxSubId = pJHg->maxSubId;

    CDPrint( cdpCD0, "%s: GPU LUT load complete in %ums", __FUNCTION__, AriocBase::aam.ms.LoadHJ );
}

/// [protected] method flushARowWriters
void AriocBase::flushARowWriters( WinGlobalPtr<baseARowWriter*>& arwList )
{
    // traverse the list of ARowWriter instances
    for( INT16 u=0; u<static_cast<INT16>(arwList.Count); ++u )
    {
        // look for non-null elements in the list
        if( arwList.p[u] )
        {
            // flush the output buffers in the uth ARowWriter in the list and close the current output file if it hasn't already been closed
            arwList.p[u]->Close();
        }
    }
}

/// [protected] method releaseGpuResources
void AriocBase::releaseGpuResources()
{
    CDPrint( cdpCD0, "%s: GPU LUT unload starts...", __FUNCTION__ );

    HiResTimer hrt(ms);

    // unload the H and J lookup tables
    tuUnloadLUT<UINT32> unloadHn( &this->Hn );
    unloadHn.Start();

    tuUnloadLUT<UINT32> unloadHg( &this->Hg );
    unloadHg.Start();

    tuUnloadLUT<UINT32> unloadJn( &this->Jn );
    unloadJn.Start();

    tuUnloadLUT<UINT32> unloadJg( &this->Jg );
    unloadJg.Start();

    // wait for the unloads to complete
    unloadHn.Wait( LUT_LOAD_TIMEOUT );
    unloadHg.Wait( LUT_LOAD_TIMEOUT );
    unloadJn.Wait( LUT_LOAD_TIMEOUT );
    unloadJg.Wait( LUT_LOAD_TIMEOUT );

    // unload the R table
    m_Rpinned.Free();

    // at this point we can reset the GPUs
    CudaDeviceBinding::ResetAvailableDevices();

    // performance metrics
    AriocBase::aam.ms.ReleasePinned = hrt.GetElapsed(false);

    CDPrint( cdpCD0, "%s: GPU LUT unload complete in %ums", __FUNCTION__, AriocBase::aam.ms.ReleasePinned );
}
#pragma endregion

#pragma region static methods
/// <summary>
/// Obtains a reference to the specified task-unit performance metrics.
/// </summary>
/// <remarks>This is a C++ static method.</remarks>
AriocTaskUnitMetrics* AriocBase::GetTaskUnitMetrics( const char* _key, const char* _baseKey )
{
    char key[64];
    if( _baseKey )
        sprintf_s( key, sizeof key, "%s_%s", _baseKey, _key );
    else
        strcpy_s( key, sizeof key, _key );

    // get a reference to the specified element in the associative array
    AriocTaskUnitMetrics* ptum = &AriocBase::tum[key];

    // if necessary, save the key string along with the metrics
    if( ptum->Key[0] == 0 )
        strcpy_s( ptum->Key, sizeof ptum->Key, key );

    // return a pointer to the metrics
    return ptum;
}
#pragma endregion
