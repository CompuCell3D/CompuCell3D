#ifndef rrOSSpecificsH
#define rrOSSpecificsH
#include "rrExporter.h"
#if defined (__MINGW32__)
#undef RR_DECLSPEC
#endif

#if defined(_MSC_VER)
#pragma warning(disable : 4996) // _CRT_SECURE_NO_WARNINGS
#pragma warning(disable : 4018) // int to unsigned int comparison
#pragma warning(disable : 4482) // prefixing enums...
#pragma warning(disable : 4251) // _CRT_SECURE_NO_WARNINGS
#pragma warning(disable : 4221) // empty cpp file

#define __FUNC__ "not defined in VS"
#define __func__ "not defined in VS"
#endif

#if defined(__CODEGEARC__)
#pragma warn -8012 			//comparing unsigned and signed
#pragma warn -8004 			//variable never used
#endif

#if defined(WIN32)
#define callConv __cdecl
#else
#define callConv
#endif

//---------------------------------------------------------------------------
// match __APPLE__ as well (from Andy S.)
#if defined (__MINGW32__) || defined(__linux) || defined (__APPLE__)
#define __FUNC__ __FUNCTION__
#endif

#ifdef _MSC_VER

#define snprintf _snprintf

//#include <stdarg.h>
//#define snprintf c99_snprintf
//
//
//RR_DECLSPEC int c99_snprintf(char* str, size_t size, const char* format, ...);
//RR_DECLSPEC int c99_vsnprintf(char* str, size_t size, const char* format, va_list ap);

#endif // _MSC_VER
#endif
