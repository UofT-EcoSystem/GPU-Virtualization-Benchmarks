/*
  CudaDeviceBinding.cpp

    Copyright (c) 2015-2019 Johns Hopkins University.  All rights reserved.

    This file is part of the Arioc software distribution.  It is subject to the license terms
    in the LICENSE.txt file found in the top-level directory of the Arioc software distribution.
    The contents of this file, in whole or in part, may only be copied, modified, propagated, or
    redistributed in accordance with the license terms contained in LICENSE.txt.
*/
#include "stdafx.h"

#pragma region static member variable definitions
#ifdef _WIN32
__declspec(thread) INT32 CudaDeviceBinding::m_tlsDeviceId = -1;
#endif
#ifdef __GNUC__
__thread INT32 CudaDeviceBinding::m_tlsDeviceId = -1;
#endif
#pragma endregion

#pragma region class CudaDeviceBindingHelper
/// nested helper class: default constructor
CudaDeviceBinding::CudaDeviceBindingHelper::CudaDeviceBindingHelper(): m_psem(NULL), m_availableDeviceMap(0x00000000)
{
    /* C++ doesn't have static constructors, so we follow a well-known pattern and implement this "helper" class
        that has a static instance (i.e. it's a singleton).  The helper-class constructor runs once
        (when getHelperInstance is first called).  The destructor runs when the C++ module is unloaded.
    */

    // count the number of CUDA-enabled devices
    int nCudaDevices = CudaCommon::GetDeviceCount();
    if( nCudaDevices == 0 )
        throw new ApplicationException( __FILE__, __LINE__, "No CUDA-enabled devices" );

    // initialize a bitmap in which each bit position represents the corresponding CUDA device ID
    m_availableDeviceMap = (1 << nCudaDevices) - 1;

    // create a semaphore
    reallocateSemaphore();

    // build an array of cudaDevicePropEx structs
    this->m_cudaDeviceProperties.Realloc( nCudaDevices, false );
    for( INT32 deviceId=0; deviceId<nCudaDevices; ++deviceId )
    {
        cudaDevicePropEx* pcdpe = this->m_cudaDeviceProperties.p + deviceId;
        CudaCommon::GetDeviceProperties( pcdpe, deviceId );
    }
}

/// nested helper class: destructor
CudaDeviceBinding::CudaDeviceBindingHelper::~CudaDeviceBindingHelper()
{
    m_availableDeviceMap = 0x00000000;
    reallocateSemaphore();
}

/// [private] method reallocateSemaphore
void CudaDeviceBinding::CudaDeviceBindingHelper::reallocateSemaphore()
{
    // discard the existing semaphore
    if( m_psem )
    {
        delete m_psem;
        m_psem = NULL;
    }

    // count the number of available CUDA devices
    int nCudaDevices = __popcnt( m_availableDeviceMap );
    if( nCudaDevices )
    {
        // create a new semaphore
        m_psem = new RaiiSemaphore( nCudaDevices, nCudaDevices );
    }
}
#pragma endregion

#pragma region constructor/destructor
/// constructor
CudaDeviceBinding::CudaDeviceBinding( DWORD msTimeout ) : m_deviceId(-1), m_deviceMutex(false)
{
    CRVALIDATOR;

    CudaDeviceBindingHelper* pcdbh = getHelperInstance();
    pcdbh->m_psem->Wait( msTimeout );

    // critical section: ensure that CPU threads have mutually-exclusive access to the bitmap of available CUDA devices
    {
        RaiiCriticalSection<CudaDeviceBinding> rcs;

        // get the 0-based position of the rightmost 1 bit (which corresponds to the CUDA device ID)
        UINT8 rval = _BitScanForward( reinterpret_cast<DWORD*>(&m_deviceId), pcdbh->m_availableDeviceMap );
        if( rval == 0 )
            throw new ApplicationException( __FILE__, __LINE__, "No available CUDA devices" );

        // zero the bit
        _bittestandreset( reinterpret_cast<LONG*>(&pcdbh->m_availableDeviceMap), m_deviceId );
    }

    // initialize the CUDA device
    CudaDeviceBinding::initializeDevice( m_deviceId );

    cudaDevicePropEx* pcdpx = CudaDeviceBinding::GetDeviceProperties( m_deviceId );
    CDPrint( cdpCD2, "%s: CUDA device %d (%s) is associated with thread 0x%08X", __FUNCTION__, m_deviceId, pcdpx->name, GetCurrentThreadId() );
}

/// destructor
CudaDeviceBinding::~CudaDeviceBinding()
{
    CRVALIDATOR;

    CudaDeviceBindingHelper* pcdbh = CudaDeviceBinding::getHelperInstance();

    // critical section: ensure that CPU threads have mutually-exclusive access to the bitmap of available CUDA devices
    {
        RaiiCriticalSection<CudaDeviceBinding> rcs;

        // set the flag in the available device bitmap
        _bittestandset( reinterpret_cast<LONG*>(&pcdbh->m_availableDeviceMap), m_deviceId );
    }

    // increment the semaphore count
    pcdbh->m_psem->Release( 1 );
}
#pragma endregion

#pragma region static methods
/// [private static] method getHelperInstance
CudaDeviceBinding::CudaDeviceBindingHelper* CudaDeviceBinding::getHelperInstance()
{
    /* a static instance of CudaDeviceBindingHelper; constructed the first time this method is called and
        destroyed when the process terminates */
    static CudaDeviceBindingHelper singleton;

    return &singleton;
}

/// [private static] method initializeDevice
void CudaDeviceBinding::initializeDevice( int deviceId )
{
    CRVALIDATOR;

#if TODO_CHOP_WHEN_DEBUGGED
    CDPrint( cdpCD0, "[%d] %s: device flags = 0x%08x...", deviceId, __FUNCTION__, DEVICE_FLAGS );
#endif

    // associate the specified CUDA device with the current CPU thread
    CRVALIDATE = cudaSetDevice( deviceId );

#if TODO_CHOP_WHEN_DEBUGGED
    for( INT32 fromDevice=0; fromDevice<32; ++fromDevice )
    {
        for( INT32 onDevice=0; onDevice<32; ++onDevice )
        {
            INT32 canAccessPeer = -1;
            cudaError_t rval = cudaDeviceCanAccessPeer( &canAccessPeer, fromDevice, onDevice );
            if( rval == 0 )
                CDPrint( cdpCD0, "%s: cudaDeviceCanAccessPeer(...,%d,%d) returns %d", __FUNCTION__, fromDevice, onDevice, canAccessPeer );
        }
    }
#endif

    // save a copy of the device ID in thread-local storage
    m_tlsDeviceId = deviceId;

#if TODO_CHOP_WHEN_DEBUGGED
    CDPrint( cdpCD0, "[%d] %s: m_tlsDeviceId = %d", deviceId, __FUNCTION__, m_tlsDeviceId );
#endif
}

/// [static] method DeviceFilter_Reset
void CudaDeviceBinding::DeviceFilter_Reset()
{
    /* remove any existing filters on the set of available CUDA devices; this makes it possible to change
        the set of available devices (by applying different filters) without restarting the process */
    CudaDeviceBindingHelper* pcdbh = getHelperInstance();
    pcdbh->m_availableDeviceMap = (1 << pcdbh->m_cudaDeviceProperties.Count) - 1;
    pcdbh->reallocateSemaphore();
}

/// [static] method DeviceFilter_MinGlobalMemory
void CudaDeviceBinding::DeviceFilter_MinGlobalMemory( size_t cbGlobalMemory )
{
    CudaDeviceBindingHelper* pcdbh = getHelperInstance();

    for( INT32 n=0; n<static_cast<INT32>(pcdbh->m_cudaDeviceProperties.Count); ++n )
    {
        // if the device has less than the specified amount of global memory, reset the corresponding bit in the available-devices bitmap
        if( pcdbh->m_cudaDeviceProperties.p[n].totalGlobalMem < cbGlobalMemory )
            _bittestandreset( reinterpret_cast<LONG*>(&pcdbh->m_availableDeviceMap), n );
    }

    pcdbh->reallocateSemaphore();
}

/// [static] method DeviceFilter_MinComputeCapability
void CudaDeviceBinding::DeviceFilter_MinComputeCapability( const char* cc )
{
    /* The CUDA "compute capability" is specified as a string formatted like this:  "M.mm",
        where M is the major part and mm is the minor part of the compute capability */
    int minMajor = 0x7FFF;
    int minMinor = 0x7FFF;
    sscanf_s( cc, "%d.%d", &minMajor, &minMinor );

    CudaDeviceBindingHelper* pcdbh = getHelperInstance();

    for( INT32 n=0; n<static_cast<INT32>(pcdbh->m_cudaDeviceProperties.Count); ++n )
    {
        // verify the major version
        bool bOk = (pcdbh->m_cudaDeviceProperties.p[n].major > minMajor);
        if( !bOk )
        {
            if( pcdbh->m_cudaDeviceProperties.p[n].major == minMajor )
            {
                // verify the minor version
                bOk = (pcdbh->m_cudaDeviceProperties.p[n].minor >= minMinor);
            }
        }

        // if the compute capability is not ok, reset the corresponding bit in the available-devices bitmap
        if( !bOk )
            _bittestandreset( reinterpret_cast<LONG*>(&pcdbh->m_availableDeviceMap), n );
    }

    pcdbh->reallocateSemaphore();
}

/// [static] method DeviceFilter_DeviceIdMap
void CudaDeviceBinding::DeviceFilter_DeviceIdMap( const UINT32 deviceIdMap )
{
    CudaDeviceBindingHelper* pcdbh = getHelperInstance();

    for( INT32 n=0; n<static_cast<INT32>(pcdbh->m_cudaDeviceProperties.Count); ++n )
        if( !_bittest( reinterpret_cast<const LONG*>(&deviceIdMap), n ) )
           _bittestandreset( reinterpret_cast<LONG*>(&pcdbh->m_availableDeviceMap), n );

    pcdbh->reallocateSemaphore();
}

/// <summary>
/// Returns the number of available CUDA devices
/// </summary>
int CudaDeviceBinding::GetAvailableDeviceCount()
{
    CudaDeviceBindingHelper* pcdbh = getHelperInstance();
    return __popcnt( pcdbh->m_availableDeviceMap );
}

/// <summary>
/// Returns a list of device properties as determined by the CUDA driver
/// </summary>
/// <param name="deviceId">CUDA device ID</param>
cudaDevicePropEx* CudaDeviceBinding::GetDeviceProperties( int deviceId )
{
    // return a reference to the cudaDevicePropEx struct for the current CUDA device
    CudaDeviceBindingHelper* pcdbh = getHelperInstance();
    return pcdbh->m_cudaDeviceProperties.p + deviceId;
}

/// <summary>
/// Initializes available CUDA device(s)
/// </summary>
void CudaDeviceBinding::InitializeAvailableDevices()
{
    CRVALIDATOR;

    // get a copy of the bitmap of available CUDA device IDs
    CudaDeviceBindingHelper* pcdbh = getHelperInstance();
    volatile DWORD dmap = pcdbh->m_availableDeviceMap;

    // traverse the set of device IDs
    DWORD deviceId = -1;
    while( _BitScanForward( &deviceId, dmap ) )
    {
        // reset the device
        CRVALIDATE = cudaSetDevice( deviceId );
        CRVALIDATE = cudaDeviceReset();

        // set device-global flags
        CRVALIDATE = cudaSetDeviceFlags( DEVICE_FLAGS );

        /* Get the initial amount of free CUDA memory.

           We don't care about the memory values, but doing this API call now seems to get the CUDA driver
            to do some kind of lazy initialization here rather than later.
        */
        size_t gmemFree = 0;
        size_t gmemTotal = 0;
        CRVALIDATE = cudaMemGetInfo( &gmemFree, &gmemTotal );

        /* Reset the bit that corresponds to the device ID.

            The variable dmap must be declared volatile or g++ will optimize it out of the loop.  But Microsoft's
            function signature is

                UINT8 _bittestandreset( LONG*, LONG )

            hence the ugly C++ casting to change the type and to cast away the volatile attribute.
        */
        _bittestandreset( const_cast<LONG*>(reinterpret_cast<volatile LONG*>(&dmap)), deviceId );
    }
}

/// <summary>
/// Resets CUDA device(s)
/// </summary>
void CudaDeviceBinding::ResetAvailableDevices()
{
    CRVALIDATOR;

    // get a copy of the bitmap of available CUDA device IDs
    CudaDeviceBindingHelper* pcdbh = getHelperInstance();
    volatile UINT32 dmap = pcdbh->m_availableDeviceMap;

    DWORD deviceId = -1;
    while( _BitScanForward( &deviceId, dmap ) )
    {
        CRVALIDATE = cudaSetDevice( deviceId ); // associate the CUDA device ID with the current CPU thread
        CRVALIDATE = cudaDeviceReset();         // reset the device

        // reset the bit that corresponds to the device ID (and see the comment in InitializeAvailableDevices above)
        _bittestandreset( const_cast<LONG*>(reinterpret_cast<volatile LONG*>(&dmap)), deviceId );
    }
}
#pragma endregion

#pragma region public methods
/// <summary>
/// Returns a list of device properties as determined by the CUDA driver
/// </summary>
cudaDevicePropEx* CudaDeviceBinding::GetDeviceProperties()
{
    // return a reference to the cudaDevicePropEx struct for the current CUDA device
    CudaDeviceBindingHelper* pcdbh = getHelperInstance();
    return pcdbh->m_cudaDeviceProperties.p + m_deviceId;
}

/// <summary>
/// Returns the CUDA device ID associated with the current CPU thread.
/// </summary>
INT32 CudaDeviceBinding::GetCudaDeviceId()
{
    /* Return the device ID from thread-local storage.

       We do this for two reasons:
       - we can verify whether the current thread is associated with a CUDA device
       - as of CUDA v6.0, cudaGetDevice seems to be a blocking call that can hang for over 1000ms if it is called
          from multiple CPU threads (perhaps because it creates new CUDA context if one does not yet exist, which
          in any event is not the behavior we want here)
    */
    if( m_tlsDeviceId < 0 )
        throw new ApplicationException( __FILE__, __LINE__, "%s: no CUDA device associated with the current thread", __FUNCTION__ );

    return m_tlsDeviceId;
}

/// <summary>
/// Returns the maximum number of bytes of CUDA global memory that can be obtained through a call to one of the CUDA memory-allocation APIs
/// </summary>
INT64 CudaDeviceBinding::GetDeviceFreeGlobalMemory()
{
    CRVALIDATOR;

    /* Get available global memory for the CUDA device associated with the current CUDA context
    
       Workaround: cudaMemGetInfo does not appear to be thread-safe (at least in CUDA v5.0), so we wrap it in a
        critical section.
    */
//    RaiiCriticalSection<CudaDeviceBinding> rcs;

    size_t gmemFree = 0;
    size_t gmemTotal = 0;
    CRVALIDATE = cudaMemGetInfo( &gmemFree, &gmemTotal );

    // return the number of bytes of available global memory
    return static_cast<INT64>(gmemFree);
}

/// <summary>
/// Obtains a mutex associated with the current CUDA device
/// </summary>
void CudaDeviceBinding::CaptureDevice( DWORD msTimeout )
{
    // get ownership of the mutex that ensures exclusive access to the CUDA device
    m_deviceMutex.Wait( msTimeout );
}

/// <summary>
/// Releases the mutex (obtained by calling <c>CaptureDevice</c>) that associated the caller with the current CUDA device
/// </summary>
void CudaDeviceBinding::ReleaseDevice()
{
    // release the mutex
    m_deviceMutex.Release();
}

/// <summary>
/// Associates the current CPU thread with the current CudaDeviceBinding device ID
/// </summary>
void CudaDeviceBinding::BindDevice()
{
    CRVALIDATOR;
    
    // associate a CUDA device with the current CPU thread
    CRVALIDATE = cudaSetDevice( m_deviceId );
}
#pragma endregion
