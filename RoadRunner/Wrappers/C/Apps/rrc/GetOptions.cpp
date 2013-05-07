#ifdef USE_PCH
#include "rr_pch.h"
#endif
#pragma hdrstop
#include <stdio.h>
#include <string.h>
#include "rrGetOptions.h"

char*    optArg = NULL;            // global argument pointer
int      optInd = 0;        // global argv index

int GetOptions(int argc, char *argv[], const char *optstring)
{
    static char *next = NULL;
    if (optInd == 0)
        next = NULL;

    optArg = NULL;

    if (next == NULL || *next == wchar_t('\0'))
    {
        if (optInd == 0)
            optInd++;

        if (optInd >= argc || argv[optInd][0] != wchar_t('-') || argv[optInd][1] == wchar_t('\0'))
        {
            optArg = NULL;
            if (optInd < argc)
                optArg = argv[optInd];
            return EOF;
        }

        if (strcmp(argv[optInd], ("--")) == 0)
        {
            optInd++;
            optArg = NULL;
            if (optInd < argc)
                optArg = argv[optInd];
            return EOF;
        }

        next = argv[optInd];
        next++;        // skip past -
        optInd++;
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
            optArg = next;
            next = NULL;
        }
        else if (optInd < argc)
        {
            optArg = argv[optInd];
            optInd++;
        }
        else
        {
            return wchar_t('?');
        }
    }

    return c;
}
