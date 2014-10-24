#ifndef CAFIELDUTILS_EXPORTS_H
#define CAFIELDUTILS_EXPORTS_H

    #if defined(_WIN32)
      #ifdef CAFieldUtilsShared_EXPORTS
          #define CAFIELDUTILS_EXPORT __declspec(dllexport)
          #define CAFIELDUTILS_EXPIMP_TEMPLATE
      #else
          #define CAFIELDUTILS_EXPORT __declspec(dllimport)
          #define CAFIELDUTILS_EXPIMP_TEMPLATE extern
      #endif
    #else
         #define CAFIELDUTILS_EXPORT
         #define CAFIELDUTILS_EXPIMP_TEMPLATE
    #endif

#endif
