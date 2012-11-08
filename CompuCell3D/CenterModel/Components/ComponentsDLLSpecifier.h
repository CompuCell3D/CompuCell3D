#ifndef COMPONENTS_EXPORT_H
#define COMPONENTS_EXPORT_H

    #if defined(_WIN32)
      #ifdef ComponentsShared_EXPORTS
          #define COMPONENTS_EXPORT __declspec(dllexport)
          #define COMPONENTS_EXPIMP_TEMPLATE
      #else
          #define COMPONENTS_EXPORT __declspec(dllimport)
          #define COMPONENTS_EXPIMP_TEMPLATE extern
      #endif
    #else
         #define COMPONENTS_EXPORT
         #define COMPONENTS_EXPIMP_TEMPLATE
    #endif

#endif
