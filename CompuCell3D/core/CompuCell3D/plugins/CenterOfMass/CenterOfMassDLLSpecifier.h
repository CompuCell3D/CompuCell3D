#ifndef CENTEROFMASS_EXPORT_H
#define CENTEROFMASS_EXPORT_H

    #if defined(_WIN32)
      #ifdef CenterOfMassShared_EXPORTS
          #define CENTEROFMASS_EXPORT __declspec(dllexport)
          #define CENTEROFMASS_EXPIMP_TEMPLATE
      #else
          #define CENTEROFMASS_EXPORT __declspec(dllimport)
          #define CENTEROFMASS_EXPIMP_TEMPLATE extern
      #endif
    #else
         #define CENTEROFMASS_EXPORT
         #define CENTEROFMASS_EXPIMP_TEMPLATE
    #endif

#endif
