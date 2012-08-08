#ifndef SIMPLECLOCK_EXPORT_H
#define SIMPLECLOCK_EXPORT_H

    #if defined(_WIN32)
      #ifdef SimpleClockShared_EXPORTS
          #define SIMPLECLOCK_EXPORT __declspec(dllexport)
          #define SIMPLECLOCK_EXPIMP_TEMPLATE
      #else
          #define SIMPLECLOCK_EXPORT __declspec(dllimport)
          #define SIMPLECLOCK_EXPIMP_TEMPLATE extern
      #endif
    #else
         #define SIMPLECLOCK_EXPORT
         #define SIMPLECLOCK_EXPIMP_TEMPLATE
    #endif

#endif
