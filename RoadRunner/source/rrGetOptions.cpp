#ifdef USE_PCH
#include "rr_pch.h"
#endif
#pragma hdrstop
#include <stdio.h>
#include <string.h>
#include "rrGetOptions.h"
//---------------------------------------------------------------------------
namespace rr
{

char    *rrOptArg;        // global argument pointer
int      rrOptInd = 0;     // global argv index

int GetOptions(int argc, char *argv[], const char* optstring)
{
    static char *next = NULL;
    if (rrOptInd == 0)
        next = NULL;

    rrOptArg = NULL;

    if (next == NULL || *next == wchar_t('\0'))
    {
        if (rrOptInd == 0)
            rrOptInd++;

        if (rrOptInd >= argc || argv[rrOptInd][0] != wchar_t('-') || argv[rrOptInd][1] == wchar_t('\0'))
        {
            rrOptArg = NULL;
            if (rrOptInd < argc)
                rrOptArg = argv[rrOptInd];
            return EOF;
        }

        if (strcmp(argv[rrOptInd], ("--")) == 0)
        {
            rrOptInd++;
            rrOptArg = NULL;
            if (rrOptInd < argc)
                rrOptArg = argv[rrOptInd];
            return EOF;
        }

        next = argv[rrOptInd];
        next++;        // skip past -
        rrOptInd++;
    }

    char c = *next++;
    const char *cp = strchr(optstring, c);

    if (cp == NULL || c == wchar_t(':'))
        return wchar_t('?');

    cp++;
    if (*cp == wchar_t(':'))
    {
        if (*next != wchar_t('\0'))
        {
            rrOptArg = next;
            next = NULL;
        }
        else if (rrOptInd < argc)
        {
            rrOptArg = argv[rrOptInd];
            rrOptInd++;
        }
        else
        {
            return wchar_t('?');
        }
    }

    return c;
}

}//namepsace
