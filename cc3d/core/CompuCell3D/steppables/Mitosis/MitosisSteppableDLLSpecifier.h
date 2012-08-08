#ifndef MITOSISSTEPPABLE_EXPORT_H
#define MITOSISSTEPPABLE_EXPORT_H

    #if defined(_WIN32)
      #ifdef MitosisSteppableShared_EXPORTS
          #define MITOSISSTEPPABLE_EXPORT __declspec(dllexport)
          #define MITOSISSTEPPABLE_EXPIMP_TEMPLATE
      #else
          #define MITOSISSTEPPABLE_EXPORT __declspec(dllimport)
          #define MITOSISSTEPPABLE_EXPIMP_TEMPLATE extern
      #endif
    #else
         #define MITOSISSTEPPABLE_EXPORT
         #define MITOSISSTEPPABLE_EXPIMP_TEMPLATE
    #endif

#endif
