#ifndef SECRETION_EXPORT_H
#define SECRETION_EXPORT_H

    #if defined(_WIN32)
      #ifdef SecretionShared_EXPORTS
          #define SECRETION_EXPORT __declspec(dllexport)
          #define SECRETION_EXPIMP_TEMPLATE
      #else
          #define SECRETION_EXPORT __declspec(dllimport)
          #define SECRETION_EXPIMP_TEMPLATE extern
      #endif
    #else
         #define SECRETION_EXPORT
         #define SECRETION_EXPIMP_TEMPLATE
    #endif

#endif
