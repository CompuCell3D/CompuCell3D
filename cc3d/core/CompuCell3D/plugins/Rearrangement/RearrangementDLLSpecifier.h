#ifndef REARRANGEMENT_EXPORT_H
#define REARRANGEMENT_EXPORT_H

    #if defined(_WIN32)
      #ifdef RearrangementShared_EXPORTS
          #define REARRANGEMENT_EXPORT __declspec(dllexport)
          #define REARRANGEMENT_EXPIMP_TEMPLATE
      #else
          #define REARRANGEMENT_EXPORT __declspec(dllimport)
          #define REARRANGEMENT_EXPIMP_TEMPLATE extern
      #endif
    #else
         #define REARRANGEMENT_EXPORT
         #define REARRANGEMENT_EXPIMP_TEMPLATE
    #endif

#endif
