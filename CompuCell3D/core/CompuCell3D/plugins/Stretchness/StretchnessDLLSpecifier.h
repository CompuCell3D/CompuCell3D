#ifndef STRETCHNESS_EXPORT_H
#define STRETCHNESS_EXPORT_H

    #if defined(_WIN32)
      #ifdef StretchnessShared_EXPORTS
          #define STRETCHNESS_EXPORT __declspec(dllexport)
          #define STRETCHNESS_EXPIMP_TEMPLATE
      #else
          #define STRETCHNESS_EXPORT __declspec(dllimport)
          #define STRETCHNESS_EXPIMP_TEMPLATE extern
      #endif
    #else
         #define STRETCHNESS_EXPORT
         #define STRETCHNESS_EXPIMP_TEMPLATE
    #endif

#endif
