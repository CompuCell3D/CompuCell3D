#ifndef XMLUTILS_EXPORT_H
#define XMLUTILS_EXPORT_H

    #if defined(_WIN32)
      #ifdef XMLUtilsShared_EXPORTS
          #define XMLUTILS_EXPORT __declspec(dllexport)
          #define XMLUTILS_EXPIMP_TEMPLATE
      #else
          #define XMLUTILS_EXPORT __declspec(dllimport)
          #define XMLUTILS_EXPIMP_TEMPLATE extern
      #endif
    #else
         #define XMLUTILS_EXPORT
         #define XMLUTILS_EXPIMP_TEMPLATE
    #endif

#endif
