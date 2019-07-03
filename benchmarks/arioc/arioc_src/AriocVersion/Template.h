/*
  AriocVersion.h (generated from Template.h)

    Copyright (c) 2015-2019 Johns Hopkins University.  All rights reserved.

    This file is part of the Arioc software distribution.  It is subject to the license terms
    in the LICENSE.txt file found in the top-level directory of the Arioc software distribution.
    The contents of this file, in whole or in part, may only be copied, modified, propagated, or
    redistributed in accordance with the license terms contained in LICENSE.txt.
*/
#ifndef __AriocVersion_Includes__
#define __AriocVersion_Includes__

/* The following are placeholders for the TortoiseSVN command-line utility SubWCrev.exe:

    - working copy revision number: $WCREV$
    - working copy "date stamp" (2-digit year, 3-digit day number): $WCDATE=%y%j$
    - current 4-digit year: $WCNOW=%Y$

    The command-line syntax is something like
        SubWCrev "C:\Projects VS120\Arioc" SvnTemplate.h AriocVersion.h
*/

#define VER_FILEVERSION             1,30,$WCREV$,$WCDATE=%y%j$
#define VER_FILEVERSION_STR         "1.30.$WCREV$.$WCDATE=%y%j$\0"

#define VER_PRODUCTVERSION          1,30,0,0
#define VER_PRODUCTVERSION_STR      "1.30\0"

#define VER_LEGALCOPYRIGHT_STR      "Copyright (c) 2015-$WCNOW=%Y$ Johns Hopkins University.  All rights reserved."
#endif