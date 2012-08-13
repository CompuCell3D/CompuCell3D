#ifndef SIMTHREAD_EXPORT_H
#define SIMTHREAD_EXPORT_H

    #if defined(_WIN32)
      #ifdef simthreadShared_EXPORTS
          #define SIMTHREAD_EXPORT __declspec(dllexport)
          #define SIMTHREAD_EXPIMP_TEMPLATE
      #else
          #define SIMTHREAD_EXPORT __declspec(dllimport)
          #define SIMTHREAD_EXPIMP_TEMPLATE extern
      #endif
    #else
         #define SIMTHREAD_EXPORT
         #define SIMTHREAD_EXPIMP_TEMPLATE
    #endif

#endif
