#ifndef CENTEROFMASSMONITOR_EXPORT_H
#define CENTEROFMASSMONITOR_EXPORT_H

    #if defined(_WIN32)
      #ifdef CenterOfMassMonitorShared_EXPORTS
          #define CENTEROFMASSMONITOR_EXPORT __declspec(dllexport)
          #define CENTEROFMASSMONITOR_EXPIMP_TEMPLATE
      #else
          #define CENTEROFMASSMONITOR_EXPORT __declspec(dllimport)
          #define CENTEROFMASSMONITOR_EXPIMP_TEMPLATE extern
      #endif
    #else
         #define CENTEROFMASSMONITOR_EXPORT
         #define CENTEROFMASSMONITOR_EXPIMP_TEMPLATE
    #endif

#endif
