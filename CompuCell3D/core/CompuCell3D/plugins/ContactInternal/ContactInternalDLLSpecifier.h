#ifndef CONTACTINTERNAL_EXPORT_H
#define CONTACTINTERNAL_EXPORT_H

    #if defined(_WIN32)
      #ifdef ContactInternalShared_EXPORTS
          #define CONTACTINTERNAL_EXPORT __declspec(dllexport)
          #define CONTACTINTERNAL_EXPIMP_TEMPLATE
      #else
          #define CONTACTINTERNAL_EXPORT __declspec(dllimport)
          #define CONTACTINTERNAL_EXPIMP_TEMPLATE extern
      #endif
    #else
         #define CONTACTINTERNAL_EXPORT
         #define CONTACTINTERNAL_EXPIMP_TEMPLATE
    #endif

#endif
