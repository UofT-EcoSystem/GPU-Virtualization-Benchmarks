/*
  CudaResult.h

    Copyright (c) 2015-2019 Johns Hopkins University.  All rights reserved.

    This file is part of the Arioc software distribution.  It is subject to the license terms
    in the LICENSE.txt file found in the top-level directory of the Arioc software distribution.
    The contents of this file, in whole or in part, may only be copied, modified, propagated, or
    redistributed in accordance with the license terms contained in LICENSE.txt.
*/
#pragma once
#define __CudaResult__

class CudaResult
{
    public:
        char    FileName[MAX_PATH];
        int     LineNumber;

    private:
#if TODO_CHOP_IF_UNUSED
        char*   getDeviceApiErrorString( CUresult );
#endif

    public:
        CudaResult( void );
        CudaResult( const char* );
        ~CudaResult( void );
        void operator=( const cudaError_t );        // runtime API error codes
#if TODO_CHOP_IF_UNUSED
        void operator=( const CUresult );           // driver API error codes
#endif
        CudaResult& SetDebugInfo( const int );
};


/* Macros to facilitate usage of this class, e.g.

    extern "C" void SomeFunction()
    {
        try
        {
            CRVALIDATOR;                            // CudaResult crInstance(__FILE__);

            CRVALIDATE = cudaFunction1( ... );      // crInstance.SetDebugInfo(__LINE__) = cudaFunction1( ... ); 
            CRVALIDATE = cudaFunction2( ... );      // crInstance.SetDebugInfo(__LINE__) = cudaFunction2( ... ); 
            ...

            CREXEC( cudaFunction3( ... ) );

        }
        catch( ApplicationException* pex )
        {
            CRTHROW;
        }
    }
*/
#define CRVALIDATOR     CudaResult crInstance( __FILE__ )
#define CRVALIDATE      crInstance.SetDebugInfo( __LINE__ )
#define CREXEC(s)       CRVALIDATE; s
#define CRTHROW         pex->SetCallerExceptionInfo(  __FILE__, crInstance.LineNumber, __FUNCTION__ );  throw pex
