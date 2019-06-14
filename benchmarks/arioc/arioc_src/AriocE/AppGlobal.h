/*
  AppGlobal.h

    Copyright (c) 2015-2019 Johns Hopkins University.  All rights reserved.

    This file is part of the Arioc software distribution.  It is subject to the license terms
    in the LICENSE.txt file found in the top-level directory of the Arioc software distribution.
    The contents of this file, in whole or in part, may only be copied, modified, propagated, or
    redistributed in accordance with the license terms contained in LICENSE.txt.
*/
#pragma once
#define __AppGlobal__

struct AppPerfMetrics
{
    INT32   nInputFiles;            // number of input files
    char    InputFileFormat[64];    // input file format
    UINT64  nSqIn;                  // total number of sequences read from the input file(s)
    UINT64  nSqEncoded;             // total number of sequences encoded
    UINT64  nSymbolsEncoded;        // total number of symbols encoded
    UINT64  nKmersIgnored;          // total number of kmers containing too many Ns to align
    volatile UINT64  nKmersUsed;    // total number of kmers encoded and hashed (i.e. total number of J values in the J table)
    INT64   nNullH;                 // number of null (unused) hash keys
    INT64   nJclamped;              // number of J lists whose length was clamped to the user-configured maximum
    INT64   nJexcluded;             // number of Jvalues excluded from the lookup table
    INT32   nConcurrentThreads;     // number of concurrent threads

    INT64   cbC;                    // number of bytes in C output file
    INT64   cbH;                    // number of bytes in H output file
    INT64   cbJ;                    // number of bytes in J output file
    INT64   cbArioc;                // number of bytes in Arioc output file(s)

    INT32   msEncoding;             // kmer encoding
    INT32   msJlistSizes;           // compute J list sizes
    INT32   msBuildJ;               // build J lists
    INT32   msSortJ;                // sort J lists
    INT64   usReadRaw;              // read raw sequence data
    INT64   usReadA21;              // read encoded sequence data
    INT32   msWriteSq;              // write raw and encoded sequence data
    INT32   msWriteKmers;           // write hashed kmers
    INT32   msWriteC;               // write J-list counts
    INT32   msWriteH;               // write the H table
    INT32   msWriteJ;               // write the J table
    INT32   msSortBB;               // sort big-bucket lists
    INT32   msCompactHtable;        // compact the H table
    INT32   msCompactJtable;        // compact the J table
    INT32   msJeol;                 // set end-of-list flags on J lists
    INT32   msValidateHJ;           // validate the contents of the H and J tables
    INT32   msValidateSubIds;       // validate the subIds in the H and J tables
    INT32   msValidateQ;            // validate the encoded Q-sequence files
    INT32   msApp;                  // overall elapsed time
};
