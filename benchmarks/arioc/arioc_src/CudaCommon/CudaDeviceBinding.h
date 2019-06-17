/*
  CudaDeviceBinding.h

    Copyright (c) 2015-2019 Johns Hopkins University.  All rights reserved.

    This file is part of the Arioc software distribution.  It is subject to the license terms
    in the LICENSE.txt file found in the top-level directory of the Arioc software distribution.
    The contents of this file, in whole or in part, may only be copied, modified, propagated, or
    redistributed in accordance with the license terms contained in LICENSE.txt.

  Notes:
   Use the nvidia-smi.exe application to set the driver model of each CUDA-capable GPU to "TCC" (as opposed to WDDM).
   In CUDA v4.0, the command looks like this:

        nvidia-smi -g <device_number> -dm 1
*/
#pragma once
#define __CudaDeviceBinding__

#define CUDADEVICEBINDING_TIMEOUT      10000    // 10 seconds

/// <summary>
/// Class <c>CudaDeviceBinding</c> implements a thread-safe exclusive-access mechanism for available CUDA devices through a static class instance
/// (or at least a C++ simulacrum of a static class).
/// </summary>
/// <remarks>This class can be used only within a single process; it does not provide an interprocess locking mechanism.
/// </remarks>
class CudaDeviceBinding
{
    static const UINT32 DEVICE_FLAGS = cudaDeviceScheduleSpin | cudaDeviceMapHost;

    /// Nested class <c>CudaDeviceBindingHelper</c> does the heavy lifting for the CudaDeviceBinding class.
    class CudaDeviceBindingHelper
    {
        /* methods in CudaDeviceBinding can access members of CudaDeviceBindingHelper as if they were themselves
            implemented in CudaDeviceBindingHelper */
        friend class CudaDeviceBinding;                     // grant CudaDeviceBinding methods access to private members of this CudaDeviceBindingHelper class

        private:
            RaiiSemaphore*                  m_psem;
            UINT32                          m_availableDeviceMap;
            WinGlobalPtr<cudaDevicePropEx>  m_cudaDeviceProperties;

        private:
            CudaDeviceBindingHelper();
            virtual ~CudaDeviceBindingHelper();
            void reallocateSemaphore( void );
    };

    private:
        int             m_deviceId;     // CUDA device ID (bit position in the available-device bitmap)
        RaiiMutex       m_deviceMutex;  // a mutex used to ensure that only one CPU thread at a time can "capture" the CUDA device

#ifdef _WIN32
        __declspec(thread) static INT32 m_tlsDeviceId;  // CUDA device ID in thread-local storage
#endif
#ifdef __GNUC__
        static __thread INT32 m_tlsDeviceId;                     // CUDA device ID in thread-local storage
#endif

    private:
        static void initializeDevice( INT32 deviceId );
        static CudaDeviceBindingHelper* getHelperInstance( void );

    public:
        CudaDeviceBinding( DWORD );
        virtual ~CudaDeviceBinding( void );
        static void DeviceFilter_Reset( void );
        static void DeviceFilter_MinGlobalMemory( size_t );
        static void DeviceFilter_MinComputeCapability( const char* );
        static void DeviceFilter_DeviceIdMap( const UINT32 );
        static INT32 GetAvailableDeviceCount( void );
        static cudaDevicePropEx* GetDeviceProperties( int deviceId );
        static void InitializeAvailableDevices( void );
        static INT32 GetCudaDeviceId( void );
        INT64 GetDeviceFreeGlobalMemory( void );
        cudaDevicePropEx* GetDeviceProperties( void );
        void CaptureDevice( DWORD msTimeout );
        void ReleaseDevice( void );
        void BindDevice( void );
        static void ResetAvailableDevices( void );
};
